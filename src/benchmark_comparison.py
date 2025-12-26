# benchmark_comparison.py
# Compare performance across different retrieval configurations
from __future__ import annotations
import argparse
import pickle
import os
import json
from typing import List, Dict, Any

from rag import retrieve, retrieve_hybrid
from benchmark import measure_query_latency, get_system_info, format_system_info


def run_comparison(
    index_configs: List[Dict[str, Any]],
    queries: List[str],
    k: int = 3,
    num_trials: int = 20
) -> Dict[str, Any]:
    """Run a comprehensive benchmark comparison.
    
    Args:
        index_configs: List of dicts with 'path', 'name', and 'methods' keys
        queries: List of query strings
        k: Number of results to retrieve
        num_trials: Number of trials per query
    
    Returns:
        Dictionary with all benchmark results
    """
    results = {
        "system_info": get_system_info(),
        "config": {
            "k": k,
            "num_trials": num_trials,
            "num_queries": len(queries)
        },
        "indexes": [],
        "comparisons": []
    }
    
    for config in index_configs:
        print(f"\nLoading index: {config['name']}")
        print(f"  Path: {config['path']}")
        
        if not os.path.exists(config['path']):
            print(f"  ⚠ Index not found, skipping")
            continue
        
        with open(config['path'], "rb") as f:
            index = pickle.load(f)
        
        index_size_mb = os.path.getsize(config['path']) / (1024 * 1024)
        
        index_info = {
            "name": config['name'],
            "path": config['path'],
            "size_mb": index_size_mb,
            "num_passages": len(index.passages),
            "has_chunks": index.chunks is not None,
            "has_bm25": index.bm25 is not None,
            "has_embeddings": index.embedder is not None,
            "has_reranker": index.reranker is not None,
            "methods": []
        }
        
        print(f"  Passages: {len(index.passages)}")
        print(f"  Size: {index_size_mb:.2f} MB")
        
        # Test each method for this index
        for method_config in config['methods']:
            method_name = method_config['name']
            method_type = method_config['method']
            rerank = method_config.get('rerank', False)
            
            print(f"\n  Testing method: {method_name}")
            
            method_results = []
            
            for query in queries:
                # Determine retrieval function
                if method_type == 'tfidf_legacy':
                    retrieve_fn = lambda idx, q, k_val: retrieve(idx, q, k=k_val)
                else:
                    retrieve_fn = lambda idx, q, k_val: retrieve_hybrid(
                        idx, q, k=k_val, method=method_type, rerank=rerank
                    )
                
                stats = measure_query_latency(
                    index, query, k, method_name, retrieve_fn, 
                    num_warmup=5, num_trials=num_trials
                )
                method_results.append(stats)
                print(f"    {query[:50]}... -> {stats['mean_ms']:.2f}ms")
            
            # Calculate aggregate stats
            avg_mean_ms = sum(r['mean_ms'] for r in method_results) / len(method_results)
            avg_p95_ms = sum(r['p95_ms'] for r in method_results) / len(method_results)
            
            index_info['methods'].append({
                "name": method_name,
                "method": method_type,
                "rerank": rerank,
                "avg_mean_latency_ms": avg_mean_ms,
                "avg_p95_latency_ms": avg_p95_ms,
                "per_query_results": method_results
            })
        
        results['indexes'].append(index_info)
    
    return results


def format_comparison_table(results: Dict[str, Any]) -> str:
    """Format comparison results as a table."""
    lines = []
    lines.append("=" * 100)
    lines.append("BENCHMARK COMPARISON")
    lines.append("=" * 100)
    lines.append(f"{'Index':<25} {'Method':<20} {'Avg Latency (ms)':<18} {'P95 (ms)':<12} {'Size (MB)':<12}")
    lines.append("-" * 100)
    
    for idx_info in results['indexes']:
        for i, method_info in enumerate(idx_info['methods']):
            index_name = idx_info['name'] if i == 0 else ""
            size_str = f"{idx_info['size_mb']:.2f}" if i == 0 else ""
            
            lines.append(
                f"{index_name:<25} "
                f"{method_info['name']:<20} "
                f"{method_info['avg_mean_latency_ms']:<18.2f} "
                f"{method_info['avg_p95_latency_ms']:<12.2f} "
                f"{size_str:<12}"
            )
    
    lines.append("=" * 100)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare performance across different retrieval configurations")
    parser.add_argument("--output", default="outputs/benchmark_comparison.json",
                       help="Output file for results")
    parser.add_argument("--num-trials", type=int, default=20,
                       help="Number of trials per query")
    parser.add_argument("--k", type=int, default=3,
                       help="Number of results to retrieve")
    args = parser.parse_args()
    
    print(format_system_info(get_system_info()))
    print()
    
    # Define test queries
    queries = [
        "What is reinforcement learning?",
        "How do you evaluate retrieval?",
        "What is TF-IDF used for?"
    ]
    
    # Define index configurations to compare
    index_configs = [
        {
            "name": "TF-IDF Only",
            "path": "outputs/index.pkl",
            "methods": [
                {"name": "tfidf", "method": "tfidf"}
            ]
        },
        {
            "name": "Chunked (TF-IDF)",
            "path": "outputs/index_chunked.pkl",
            "methods": [
                {"name": "tfidf", "method": "tfidf"}
            ]
        },
        {
            "name": "Hybrid",
            "path": "outputs/index_hybrid.pkl",
            "methods": [
                {"name": "tfidf", "method": "tfidf"},
                {"name": "bm25", "method": "bm25"},
                {"name": "embeddings", "method": "embeddings"},
                {"name": "hybrid", "method": "hybrid"}
            ]
        },
        {
            "name": "Hybrid + Chunking",
            "path": "outputs/index_chunked_hybrid.pkl",
            "methods": [
                {"name": "tfidf", "method": "tfidf"},
                {"name": "embeddings", "method": "embeddings"},
                {"name": "hybrid", "method": "hybrid"}
            ]
        }
    ]
    
    print(f"Running benchmark comparison with {len(queries)} queries, {args.num_trials} trials each")
    print("=" * 100)
    
    results = run_comparison(index_configs, queries, k=args.k, num_trials=args.num_trials)
    
    # Print comparison table
    print("\n")
    print(format_comparison_table(results))
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {args.output}")
    
    # Print key insights
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)
    
    if results['indexes']:
        # Find fastest and slowest
        all_methods = [
            (idx['name'], method)
            for idx in results['indexes']
            for method in idx['methods']
        ]
        
        fastest = min(all_methods, key=lambda x: x[1]['avg_mean_latency_ms'])
        slowest = max(all_methods, key=lambda x: x[1]['avg_mean_latency_ms'])
        
        print(f"Fastest: {fastest[0]} / {fastest[1]['name']} - {fastest[1]['avg_mean_latency_ms']:.2f}ms avg")
        print(f"Slowest: {slowest[0]} / {slowest[1]['name']} - {slowest[1]['avg_mean_latency_ms']:.2f}ms avg")
        
        # Handle division by zero for very fast methods
        if fastest[1]['avg_mean_latency_ms'] > 0:
            speed_diff = slowest[1]['avg_mean_latency_ms'] / fastest[1]['avg_mean_latency_ms']
            print(f"Speed difference: {speed_diff:.1f}x")
        else:
            print(f"Speed difference: >1000x (fastest method < 0.01ms)")
        
        # Index size comparison
        smallest_idx = min(results['indexes'], key=lambda x: x['size_mb'])
        largest_idx = max(results['indexes'], key=lambda x: x['size_mb'])
        
        print(f"\nSmallest index: {smallest_idx['name']} - {smallest_idx['size_mb']:.2f} MB")
        print(f"Largest index: {largest_idx['name']} - {largest_idx['size_mb']:.2f} MB")
        print(f"Size ratio: {largest_idx['size_mb'] / smallest_idx['size_mb']:.1f}x")
    
    print("=" * 100)


if __name__ == "__main__":
    main()
