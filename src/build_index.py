# build_index.py
# This script builds an index from a text file and saves it to a pickle file.
from __future__ import annotations

import argparse
import os
import pickle
import time

from benchmark import Benchmark, format_system_info, get_system_info
from io_utils import read_text
from rag import build_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="data/docs.txt")
    parser.add_argument("--out", default="outputs/index.pkl")
    parser.add_argument("--bm25", action="store_true", help="Include BM25 for hybrid retrieval")
    parser.add_argument("--embeddings", action="store_true", help="Include dense embeddings")
    parser.add_argument("--reranker", action="store_true", help="Load cross-encoder reranker")
    parser.add_argument(
        "--chunking",
        action="store_true",
        help="Enable chunking with overlap for grounded retrieval",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=200, help="Chunk size in characters (default: 200)"
    )
    parser.add_argument(
        "--overlap", type=int, default=50, help="Overlap between chunks in characters (default: 50)"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Show detailed performance metrics"
    )
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    if args.benchmark:
        print(format_system_info(get_system_info()))
        print()

    print("Building index...")
    if args.bm25:
        print("  BM25 enabled")
    if args.embeddings:
        print("  Dense embeddings enabled")
    if args.reranker:
        print("  Reranker enabled")
    if args.chunking:
        print(f"  Chunking enabled (size={args.chunk_size}, overlap={args.overlap})")

    # Read corpus
    read_start = time.time()
    corpus = read_text(args.docs)
    read_time = time.time() - read_start

    # Build index with benchmarking
    with Benchmark("Index Building", track_memory=True) as bench:
        index = build_index(
            corpus,
            use_bm25=args.bm25,
            use_embeddings=args.embeddings,
            use_reranker=args.reranker,
            use_chunking=args.chunking,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )

    # Save index
    save_start = time.time()
    with open(args.out, "wb") as f:
        pickle.dump(index, f)
    save_time = time.time() - save_start

    # Get file size
    index_size_mb = os.path.getsize(args.out) / (1024 * 1024)

    print(f"\n{'='*70}")
    print("INDEX BUILD COMPLETE")
    print(f"{'='*70}")
    print(f"Output: {args.out}")
    print(f"Passages/Chunks: {len(index.passages)}")
    if index.chunks:
        num_source_docs = len(set(c.source_doc_id for c in index.chunks))
        print(f"Total chunks: {len(index.chunks)} (from {num_source_docs} source documents)")

    if args.benchmark:
        result = bench.get_result(
            num_passages=len(index.passages),
            has_bm25=index.bm25 is not None,
            has_embeddings=index.embedder is not None,
            has_reranker=index.reranker is not None,
            has_chunking=index.chunks is not None,
        )

        print(f"\n{'='*70}")
        print("PERFORMANCE METRICS")
        print(f"{'='*70}")
        print(f"Read corpus time: {read_time:.3f}s")
        print(f"Build time: {result.wall_time:.3f}s")
        print(f"Save time: {save_time:.3f}s")
        print(f"Total time: {read_time + result.wall_time + save_time:.3f}s")
        print("\nMemory:")
        if result.memory_used_mb is not None:
            print(f"  Memory used: {result.memory_used_mb:.2f} MB")
        if result.peak_memory_mb is not None:
            print(f"  Peak memory: {result.peak_memory_mb:.2f} MB")
        print(f"\nIndex size on disk: {index_size_mb:.2f} MB")

        if len(index.passages) > 0:
            passages_per_sec = len(index.passages) / result.wall_time
            print(f"Indexing throughput: {passages_per_sec:.2f} passages/sec")

        print(f"{'='*70}")


if __name__ == "__main__":
    main()
