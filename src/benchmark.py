# benchmark.py
# Performance measurement utilities for latency, memory, and throughput analysis.
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import psutil


@dataclass
class BenchmarkResult:
    """Container for benchmark measurements."""

    operation: str
    wall_time: float  # seconds
    cpu_time: float | None = None  # seconds
    memory_used_mb: float | None = None
    peak_memory_mb: float | None = None
    throughput: float | None = None  # items/second
    metadata: dict[str, Any] = field(default_factory=dict)

    def format_summary(self) -> str:
        """Format benchmark results as a readable summary."""
        lines = [f"Operation: {self.operation}"]
        lines.append(f"  Wall time: {self.wall_time:.3f}s")

        if self.cpu_time is not None:
            lines.append(f"  CPU time: {self.cpu_time:.3f}s")

        if self.memory_used_mb is not None:
            lines.append(f"  Memory used: {self.memory_used_mb:.2f} MB")

        if self.peak_memory_mb is not None:
            lines.append(f"  Peak memory: {self.peak_memory_mb:.2f} MB")

        if self.throughput is not None:
            lines.append(f"  Throughput: {self.throughput:.2f} items/sec")

        for key, value in self.metadata.items():
            lines.append(f"  {key}: {value}")

        return "\n".join(lines)


class Benchmark:
    """Context manager for measuring performance metrics."""

    def __init__(self, operation: str, track_memory: bool = True):
        self.operation = operation
        self.track_memory = track_memory
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.peak_memory = None
        self.process = psutil.Process(os.getpid()) if track_memory else None

    def __enter__(self):
        self.start_time = time.time()
        if self.track_memory and self.process:
            self.start_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
            self.peak_memory = self.start_memory
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        return False

    def get_result(self, throughput_count: int | None = None, **metadata) -> BenchmarkResult:
        """Get benchmark results."""
        wall_time = self.end_time - self.start_time if self.end_time else 0.0

        memory_used = None
        peak_memory = None
        if self.track_memory and self.process:
            current_memory = self.process.memory_info().rss / (1024 * 1024)
            memory_used = current_memory - self.start_memory
            peak_memory = current_memory

        throughput = None
        if throughput_count is not None and wall_time > 0:
            throughput = throughput_count / wall_time

        return BenchmarkResult(
            operation=self.operation,
            wall_time=wall_time,
            memory_used_mb=memory_used,
            peak_memory_mb=peak_memory,
            throughput=throughput,
            metadata=metadata,
        )


@contextmanager
def benchmark(operation: str, track_memory: bool = True):
    """Context manager for quick benchmarking.

    Usage:
        with benchmark("My Operation") as b:
            # do work
            pass
        result = b.get_result()
    """
    bench = Benchmark(operation, track_memory=track_memory)
    with bench:
        yield bench


def measure_query_latency(
    index: Any,
    query: str,
    k: int,
    method: str,
    retrieve_fn: callable,
    num_warmup: int = 5,
    num_trials: int = 20,
) -> dict[str, Any]:
    """Measure query latency with warmup and multiple trials.

    Args:
        index: The retrieval index
        query: Query string
        k: Number of results
        method: Retrieval method
        retrieve_fn: Function to call for retrieval (signature: fn(index, query, k))
        num_warmup: Number of warmup queries (not measured)
        num_trials: Number of measured trials

    Returns:
        Dictionary with latency statistics
    """
    # Warmup
    for _ in range(num_warmup):
        retrieve_fn(index, query, k)

    # Measure
    latencies = []
    for _ in range(num_trials):
        start = time.time()
        retrieve_fn(index, query, k)
        latencies.append(time.time() - start)

    latencies.sort()

    return {
        "method": method,
        "query": query,
        "k": k,
        "num_trials": num_trials,
        "mean_ms": sum(latencies) / len(latencies) * 1000,
        "median_ms": latencies[len(latencies) // 2] * 1000,
        "p95_ms": latencies[int(len(latencies) * 0.95)] * 1000,
        "p99_ms": latencies[int(len(latencies) * 0.99)] * 1000,
        "min_ms": min(latencies) * 1000,
        "max_ms": max(latencies) * 1000,
    }


def format_latency_table(stats: list[dict[str, Any]]) -> str:
    """Format latency statistics as a table."""
    lines = []
    lines.append("=" * 80)
    lines.append("LATENCY BENCHMARK")
    lines.append("=" * 80)
    lines.append(
        f"{'Method':<15} {'Mean (ms)':<12} {'Median (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12}"
    )
    lines.append("-" * 80)

    for stat in stats:
        lines.append(
            f"{stat['method']:<15} "
            f"{stat['mean_ms']:<12.2f} "
            f"{stat['median_ms']:<12.2f} "
            f"{stat['p95_ms']:<12.2f} "
            f"{stat['p99_ms']:<12.2f}"
        )

    lines.append("=" * 80)
    return "\n".join(lines)


def get_system_info() -> dict[str, Any]:
    """Get system information for benchmark context."""
    return {
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "total_memory_gb": psutil.virtual_memory().total / (1024**3),
        "available_memory_gb": psutil.virtual_memory().available / (1024**3),
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
    }


def format_system_info(info: dict[str, Any]) -> str:
    """Format system information for display."""
    lines = []
    lines.append("System Information:")
    lines.append(f"  CPU cores: {info['cpu_count']} physical, {info['cpu_count_logical']} logical")
    lines.append(f"  Total memory: {info['total_memory_gb']:.2f} GB")
    lines.append(f"  Available memory: {info['available_memory_gb']:.2f} GB")
    lines.append(f"  Python version: {info['python_version']}")
    return "\n".join(lines)


def generate_markdown_report(
    benchmark_results: dict[str, Any],
    output_file: str = "outputs/benchmark_report.md"
) -> None:
    """
    Generate a comprehensive Markdown benchmark report.

    Args:
        benchmark_results: Dictionary containing benchmark data
        output_file: Path to output markdown file
    """
    import datetime

    lines = []
    lines.append("# RAG-Lite Benchmark Report")
    lines.append(f"\n**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # System information
    if "system_info" in benchmark_results:
        info = benchmark_results["system_info"]
        lines.append("## System Configuration\n")
        lines.append(f"- **CPU:** {info.get('cpu_count', 'N/A')} physical cores, {info.get('cpu_count_logical', 'N/A')} logical")
        lines.append(f"- **Memory:** {info.get('total_memory_gb', 0):.2f} GB total")
        lines.append(f"- **Python:** {info.get('python_version', 'N/A')}")
        lines.append("")

    # Summary table
    lines.append("## Performance Summary\n")
    lines.append("| Method | Mean Latency (ms) | P95 (ms) | P99 (ms) | Memory (MB) | QPS |")
    lines.append("|--------|-------------------|----------|----------|-------------|-----|")

    for method_name, data in benchmark_results.items():
        if method_name == "system_info":
            continue

        latency = data.get("latency", {})
        memory = data.get("memory", {})
        throughput = data.get("throughput", {})

        mean = latency.get("mean", 0) * 1000  # Convert to ms
        p95 = latency.get("p95", 0) * 1000
        p99 = latency.get("p99", 0) * 1000
        mem = memory.get("peak_mb", 0)
        qps = throughput.get("queries_per_second", 0)

        lines.append(f"| {method_name} | {mean:.2f} | {p95:.2f} | {p99:.2f} | {mem:.2f} | {qps:.2f} |")

    lines.append("")

    # Detailed metrics
    lines.append("## Detailed Metrics\n")

    for method_name, data in benchmark_results.items():
        if method_name == "system_info":
            continue

        lines.append(f"### {method_name.upper()}\n")

        # Latency
        if "latency" in data:
            latency = data["latency"]
            lines.append("**Latency Statistics:**\n")
            lines.append(f"- Mean: {latency.get('mean', 0)*1000:.4f} ms")
            lines.append(f"- Median: {latency.get('median', 0)*1000:.4f} ms")
            lines.append(f"- P95: {latency.get('p95', 0)*1000:.4f} ms")
            lines.append(f"- P99: {latency.get('p99', 0)*1000:.4f} ms")
            lines.append(f"- Min: {latency.get('min', 0)*1000:.4f} ms")
            lines.append(f"- Max: {latency.get('max', 0)*1000:.4f} ms")
            lines.append("")

        # Memory
        if "memory" in data:
            memory = data["memory"]
            lines.append("**Memory Usage:**\n")
            lines.append(f"- Peak: {memory.get('peak_mb', 0):.2f} MB")
            lines.append(f"- Average: {memory.get('avg_mb', 0):.2f} MB")
            lines.append("")

        # Throughput
        if "throughput" in data:
            throughput = data["throughput"]
            lines.append("**Throughput:**\n")
            lines.append(f"- Queries per second: {throughput.get('queries_per_second', 0):.2f}")
            if "passages_per_second" in throughput:
                lines.append(f"- Passages per second: {throughput.get('passages_per_second', 0):.2f}")
            lines.append("")

        # Build time
        if "build_time_seconds" in data:
            lines.append(f"**Index Build Time:** {data['build_time_seconds']:.2f}s\n")

        lines.append("---\n")

    # Recommendations
    lines.append("## Recommendations\n")
    lines.append("### Method Selection Guide\n")
    lines.append("- **TF-IDF:** Fast baseline, lowest memory footprint, good for keyword matching")
    lines.append("- **BM25:** Better ranking than TF-IDF, minimal overhead, recommended baseline")
    lines.append("- **Embeddings:** Semantic search capability, higher latency and memory cost")
    lines.append("- **Hybrid:** Best quality, combines lexical + semantic, moderate overhead")
    lines.append("- **Reranking:** Highest quality, significant latency cost, use for <100 candidates\n")

    # Write to file
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"✓ Markdown report saved to: {output_file}")


def generate_json_report(
    benchmark_results: dict[str, Any],
    output_file: str = "outputs/benchmark_report.json"
) -> None:
    """
    Generate a JSON benchmark report.

    Args:
        benchmark_results: Dictionary containing benchmark data
        output_file: Path to output JSON file
    """
    import json
    import datetime

    report = {
        "generated_at": datetime.datetime.now().isoformat(),
        "results": benchmark_results
    }

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"✓ JSON report saved to: {output_file}")


def benchmark_all_methods(
    passages: list[str],
    trials: int = 10,
    k: int = 5,
    use_embeddings: bool = True,
    use_reranker: bool = True,
) -> dict[str, Any]:
    """
    Run comprehensive benchmarks across all retrieval methods.

    Args:
        passages: List of text passages to index
        trials: Number of trials for latency measurement
        k: Number of results to retrieve
        use_embeddings: Whether to test embedding-based methods
        use_reranker: Whether to test reranking

    Returns:
        Dictionary containing benchmark results for all methods
    """
    from src.rag import build_index, retrieve_hybrid
    import statistics

    results = {
        "system_info": {
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        }
    }

    test_queries = [
        "machine learning",
        "neural networks",
        "deep learning algorithms",
    ]

    # Build indices for each method
    print("\n📦 Building indices...")

    # TF-IDF
    print("  Building TF-IDF index...")
    with benchmark("build_tfidf", track_memory=True) as b:
        idx_tfidf = build_index(
            passages,
            use_bm25=False,
            use_embeddings=False,
            use_reranker=False,
            use_chunking=False,
        )
    result_tfidf = b.get_result()

    # BM25
    print("  Building BM25 index...")
    with benchmark("build_bm25", track_memory=True) as b:
        idx_bm25 = build_index(
            passages,
            use_bm25=True,
            use_embeddings=False,
            use_reranker=False,
            use_chunking=False,
        )
    result_bm25 = b.get_result()

    # Embeddings (if enabled)
    idx_embeddings = None
    result_embeddings = None
    if use_embeddings:
        print("  Building embeddings index...")
        with benchmark("build_embeddings", track_memory=True) as b:
            idx_embeddings = build_index(
                passages,
                use_bm25=False,
                use_embeddings=True,
                use_reranker=False,
                use_chunking=False,
            )
        result_embeddings = b.get_result()

    # Run query benchmarks
    print(f"\n🔍 Running query benchmarks ({trials} trials per method)...")

    methods_to_test = {
        "tfidf": idx_tfidf,
        "bm25": idx_bm25,
    }

    if use_embeddings and idx_embeddings:
        methods_to_test["embeddings"] = idx_embeddings
        methods_to_test["hybrid"] = build_index(
            passages,
            use_bm25=True,
            use_embeddings=True,
            use_reranker=False,
            use_chunking=False,
        )

    for method_name, index in methods_to_test.items():
        print(f"  Testing {method_name}...")
        latencies = []
        memories = []

        for query in test_queries:
            # Warmup
            for _ in range(2):
                retrieve_hybrid(index, query, k, method=method_name if method_name != "hybrid" else "hybrid")

            # Measure
            for _ in range(trials):
                mem_before = psutil.Process().memory_info().rss / (1024**2)
                start = time.time()
                retrieve_hybrid(index, query, k, method=method_name if method_name != "hybrid" else "hybrid")
                latencies.append(time.time() - start)
                mem_after = psutil.Process().memory_info().rss / (1024**2)
                memories.append(mem_after - mem_before)

        # Calculate statistics
        latencies.sort()
        results[method_name] = {
            "latency": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "p95": latencies[int(len(latencies) * 0.95)] if latencies else 0,
                "p99": latencies[int(len(latencies) * 0.99)] if latencies else 0,
            },
            "memory": {
                "peak_mb": max(memories) if memories else 0,
                "avg_mb": statistics.mean(memories) if memories else 0,
            },
            "throughput": {
                "queries_per_second": 1.0 / statistics.mean(latencies) if latencies and statistics.mean(latencies) > 0 else 0,
            }
        }

        # Add build times
        if method_name == "tfidf":
            results[method_name]["build_time_seconds"] = result_tfidf.wall_time
        elif method_name == "bm25":
            results[method_name]["build_time_seconds"] = result_bm25.wall_time
        elif method_name == "embeddings" and result_embeddings:
            results[method_name]["build_time_seconds"] = result_embeddings.wall_time

    return results
