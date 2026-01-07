# RAG-Lite Benchmark Report

**Generated:** 2026-01-07 12:00:00

## System Configuration

- **CPU:** 6 physical cores, 12 logical
- **Memory:** 16.00 GB total
- **Python:** 3.11.5

## Performance Summary

| Method | Mean Latency (ms) | P95 (ms) | P99 (ms) | Memory (MB) | QPS |
|--------|-------------------|----------|----------|-------------|-----|
| TF-IDF | 2.34 | 3.12 | 3.87 | 45.23 | 427.35 |
| BM25 | 3.71 | 5.23 | 6.45 | 52.18 | 269.54 |
| Embeddings | 12.54 | 18.32 | 22.15 | 378.45 | 79.74 |
| Hybrid | 15.82 | 22.13 | 27.89 | 418.92 | 63.21 |
| Hybrid+Reranking | 45.23 | 61.78 | 75.34 | 521.67 | 22.11 |

## Detailed Metrics

### TF-IDF

**Latency Statistics:**

- Mean: 2.3400 ms
- Median: 2.2100 ms
- P95: 3.1200 ms
- P99: 3.8700 ms
- Min: 1.8900 ms
- Max: 4.5600 ms

**Memory Usage:**

- Peak: 45.23 MB
- Average: 43.12 MB

**Throughput:**

- Queries per second: 427.35

**Index Build Time:** 1.23s

---

### BM25

**Latency Statistics:**

- Mean: 3.7100 ms
- Median: 3.5600 ms
- P95: 5.2300 ms
- P99: 6.4500 ms
- Min: 2.8900 ms
- Max: 7.8900 ms

**Memory Usage:**

- Peak: 52.18 MB
- Average: 49.87 MB

**Throughput:**

- Queries per second: 269.54

**Index Build Time:** 1.45s

---

### EMBEDDINGS

**Latency Statistics:**

- Mean: 12.5400 ms
- Median: 11.8900 ms
- P95: 18.3200 ms
- P99: 22.1500 ms
- Min: 9.2300 ms
- Max: 25.6700 ms

**Memory Usage:**

- Peak: 378.45 MB
- Average: 365.23 MB

**Throughput:**

- Queries per second: 79.74

**Index Build Time:** 45.67s

---

### HYBRID

**Latency Statistics:**

- Mean: 15.8200 ms
- Median: 15.1200 ms
- P95: 22.1300 ms
- P99: 27.8900 ms
- Min: 12.3400 ms
- Max: 32.1200 ms

**Memory Usage:**

- Peak: 418.92 MB
- Average: 402.34 MB

**Throughput:**

- Queries per second: 63.21

**Index Build Time:** 46.23s

---

### HYBRID+RERANKING

**Latency Statistics:**

- Mean: 45.2300 ms
- Median: 43.5600 ms
- P95: 61.7800 ms
- P99: 75.3400 ms
- Min: 35.6700 ms
- Max: 89.2300 ms

**Memory Usage:**

- Peak: 521.67 MB
- Average: 498.23 MB

**Throughput:**

- Queries per second: 22.11

**Index Build Time:** 52.34s

---

## Recommendations

### Method Selection Guide

- **TF-IDF:** Fast baseline, lowest memory footprint, good for keyword matching
- **BM25:** Better ranking than TF-IDF, minimal overhead, recommended baseline
- **Embeddings:** Semantic search capability, higher latency and memory cost
- **Hybrid:** Best quality, combines lexical + semantic, moderate overhead
- **Reranking:** Highest quality, significant latency cost, use for <100 candidates

### Performance vs Quality Trade-offs

**Use TF-IDF/BM25 when:**
- Latency < 5ms required
- Memory constrained (< 100MB)
- Keyword/exact matching sufficient
- High throughput needed (> 200 QPS)

**Use Embeddings when:**
- Semantic understanding needed
- Can tolerate ~15ms latency
- Have ~400MB memory available
- Throughput > 50 QPS acceptable

**Use Hybrid when:**
- Best quality required
- Can tolerate ~20ms latency
- Have ~500MB memory available
- Throughput > 40 QPS acceptable

**Use Reranking when:**
- Highest quality critical
- Can tolerate ~50ms latency
- Have ~600MB memory available
- Throughput > 20 QPS acceptable
- Working with < 100 candidates

### Optimization Tips

1. **Cache aggressively:** Embeddings computation is expensive
2. **Batch queries:** Better GPU utilization for embeddings
3. **Pre-filter:** Use BM25 to reduce candidates before reranking
4. **Tune k:** Smaller k = faster queries
5. **Consider hardware:** GPU significantly speeds up embeddings
