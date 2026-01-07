"""
Unified command-line interface for RAG-Lite.

Provides commands for building indices, querying, evaluation, and benchmarking.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import psutil

from src.benchmark import (
    Benchmark,
    format_system_info,
    get_system_info,
)
from src.io_utils import read_text
from src.rag import build_index, retrieve_grounded, retrieve_hybrid


def cmd_build(args: argparse.Namespace) -> None:
    """Build an index from documents."""
    os.makedirs("outputs", exist_ok=True)

    if args.verbose:
        print(format_system_info(get_system_info()))
        print()

    print(f"Building index from: {args.docs}")
    if args.bm25:
        print("  ✓ BM25 enabled")
    if args.embeddings:
        print("  ✓ Dense embeddings enabled")
    if args.reranker:
        print("  ✓ Reranker enabled")
    if args.chunking:
        print(f"  ✓ Chunking enabled (size={args.chunk_size}, overlap={args.overlap})")

    passages = read_text(args.docs)

    if args.verbose:
        benchmark = Benchmark("Index Build", track_memory=True)
        benchmark.start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024 * 1024)

    index = build_index(
        passages,
        use_bm25=args.bm25,
        use_embeddings=args.embeddings,
        use_reranker=args.reranker,
        use_chunking=args.chunking,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    if args.verbose:
        build_time = time.time() - benchmark.start_time
        end_memory = process.memory_info().rss / (1024 * 1024)
        memory_used = end_memory - start_memory
        print(f"\nBuild time: {build_time:.2f}s")
        print(f"Memory used: {memory_used:.2f} MB")
        print(f"Passages indexed: {len(passages)}")
        if index.chunks:
            print(f"Chunks created: {len(index.chunks)}")

    with open(args.output, "wb") as f:
        pickle.dump(index, f)

    print(f"\n✓ Index saved to: {args.output}")


def cmd_query(args: argparse.Namespace) -> None:
    """Execute a query against the index."""
    if not os.path.exists(args.index):
        print(f"✗ Error: Index not found: {args.index}", file=sys.stderr)
        print("  Run 'rag-lite build' first to create an index.", file=sys.stderr)
        sys.exit(1)

    with open(args.index, "rb") as f:
        index = pickle.load(f)

    # Determine retrieval function
    if args.grounded:
        results = retrieve_grounded(
            index,
            args.query,
            k=args.k,
            method=args.method,
            rerank=args.rerank,
            snippet_length=args.snippet_length,
        )
    else:
        # Use retrieve_hybrid which supports all methods
        idx_score_text = retrieve_hybrid(
            index,
            args.query,
            k=args.k,
            method=args.method,
            rerank=args.rerank
        )
        # Convert to (text, score) format for consistency
        results = [(text, score) for idx, score, text in idx_score_text]

    # Display results
    if args.json:
        if args.grounded:
            output = [
                {
                    "rank": r.rank,
                    "citation": r.citation,
                    "text": r.text,
                    "snippet": r.snippet,
                    "score": r.score,
                    "chunk_id": r.chunk_id,
                    "source_doc_id": r.source_doc_id,
                }
                for r in results
            ]
        else:
            output = [{"rank": i + 1, "text": text, "score": score} for i, (text, score) in enumerate(results)]
        print(json.dumps(output, indent=2))
    else:
        print(f"\n🔍 Query: {args.query}")
        print(f"Method: {args.method}")
        print(f"Top {args.k} results:\n")

        for i, result in enumerate(results, 1):
            if args.grounded:
                # GroundedResult object
                print(f"{i}. {result.citation}")
                print(f"   [Score: {result.score:.4f}]")
                print(f"   {result.snippet}")
            else:
                text, score = result
                print(f"{i}. [Score: {score:.4f}]")
                print(f"   {text[:200]}...")
            print()


def cmd_eval(args: argparse.Namespace) -> None:
    """Run evaluation on test queries."""
    from src.evaluate import evaluate_retrieval

    if not os.path.exists(args.index):
        print(f"✗ Error: Index not found: {args.index}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.eval_file):
        print(f"✗ Error: Evaluation file not found: {args.eval_file}", file=sys.stderr)
        sys.exit(1)

    with open(args.index, "rb") as f:
        index = pickle.load(f)

    print(f"Running evaluation with {args.method} (k={args.k})...")

    metrics = evaluate_retrieval(
        index,
        args.eval_file,
        k=args.k,
        method=args.method,
        rerank=args.rerank,
        output_file=args.output,
    )

    print("\n📊 Evaluation Results:")
    print(f"  MRR@{args.k}: {metrics['mrr']:.4f}")
    print(f"  nDCG@{args.k}: {metrics['ndcg']:.4f}")
    print(f"  Precision@{args.k}: {metrics['precision']:.4f}")
    print(f"  Recall@{args.k}: {metrics['recall']:.4f}")

    if args.output:
        print(f"\n✓ Detailed report saved to: {args.output}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run performance benchmarks."""
    from src.benchmark import benchmark_all_methods

    if not os.path.exists(args.docs):
        print(f"✗ Error: Document file not found: {args.docs}", file=sys.stderr)
        sys.exit(1)

    passages = read_text(args.docs)

    print("Running comprehensive benchmarks...")
    print(f"Documents: {len(passages)}")
    print(f"Trials: {args.trials}\n")

    results = benchmark_all_methods(
        passages,
        trials=args.trials,
        k=args.k,
        use_embeddings=not args.no_embeddings,
        use_reranker=not args.no_reranking,
    )

    # Output results
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80 + "\n")
        for method, data in results.items():
            if method == "system_info":
                continue
            print(f"{method.upper()}")
            if 'latency' in data:
                print(f"  Mean latency: {data['latency']['mean']:.4f}s")
                print(f"  P95 latency: {data['latency']['p95']:.4f}s")
            if 'memory' in data:
                print(f"  Memory: {data['memory']['peak_mb']:.2f} MB")
            print()

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="rag-lite",
        description="Production-grade retrieval system with multiple methods and benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build an index from documents")
    build_parser.add_argument("--docs", default="data/docs.txt", help="Path to documents file")
    build_parser.add_argument("--output", "-o", default="outputs/index.pkl", help="Output index file")
    build_parser.add_argument("--bm25", action="store_true", help="Enable BM25")
    build_parser.add_argument("--embeddings", action="store_true", help="Enable dense embeddings")
    build_parser.add_argument("--reranker", action="store_true", help="Load reranker model")
    build_parser.add_argument("--chunking", action="store_true", help="Enable document chunking")
    build_parser.add_argument("--chunk-size", type=int, default=200, help="Chunk size in characters")
    build_parser.add_argument("--overlap", type=int, default=50, help="Overlap between chunks")
    build_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    build_parser.set_defaults(func=cmd_build)

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the index")
    query_parser.add_argument("query", help="Query string")
    query_parser.add_argument("--index", "-i", default="outputs/index.pkl", help="Index file")
    query_parser.add_argument("--k", type=int, default=3, help="Number of results")
    query_parser.add_argument(
        "--method",
        choices=["tfidf", "bm25", "embeddings", "hybrid"],
        default="tfidf",
        help="Retrieval method",
    )
    query_parser.add_argument("--rerank", action="store_true", help="Enable reranking")
    query_parser.add_argument("--grounded", action="store_true", help="Show grounded results with citations")
    query_parser.add_argument("--snippet-length", type=int, default=150, help="Snippet length")
    query_parser.add_argument("--json", action="store_true", help="Output as JSON")
    query_parser.set_defaults(func=cmd_query)

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate retrieval performance")
    eval_parser.add_argument("--index", "-i", default="outputs/index.pkl", help="Index file")
    eval_parser.add_argument("--eval-file", default="data/eval.jsonl", help="Evaluation dataset")
    eval_parser.add_argument("--k", type=int, default=10, help="Number of results")
    eval_parser.add_argument(
        "--method",
        choices=["tfidf", "bm25", "embeddings", "hybrid"],
        default="tfidf",
        help="Retrieval method",
    )
    eval_parser.add_argument("--rerank", action="store_true", help="Enable reranking")
    eval_parser.add_argument("--output", "-o", help="Output report file")
    eval_parser.set_defaults(func=cmd_eval)

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    bench_parser.add_argument("--docs", default="data/docs.txt", help="Documents file")
    bench_parser.add_argument("--trials", type=int, default=20, help="Number of trials")
    bench_parser.add_argument("--k", type=int, default=10, help="Number of results")
    bench_parser.add_argument("--no-embeddings", action="store_true", help="Skip embeddings")
    bench_parser.add_argument("--no-reranking", action="store_true", help="Skip reranking")
    bench_parser.add_argument("--output", "-o", help="Output JSON file")
    bench_parser.add_argument("--json", action="store_true", help="Output as JSON")
    bench_parser.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
