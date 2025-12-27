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
