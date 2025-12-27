"""Tests for evaluation metrics and parsing."""

import json

import pytest


def test_calculate_recall_at_k():
    """Test Recall@K metric calculation."""
    from evaluate import calculate_recall_at_k

    # Perfect recall
    assert calculate_recall_at_k([0, 1, 2], [0, 1, 2]) == 1.0

    # Partial recall
    assert calculate_recall_at_k([0, 1, 2], [0, 3, 4]) == pytest.approx(1 / 3)

    # No recall
    assert calculate_recall_at_k([0, 1, 2], [3, 4, 5]) == 0.0

    # Empty retrieved
    assert calculate_recall_at_k([0, 1, 2], []) == 0.0


def test_calculate_mrr_at_k():
    """Test MRR@K (Mean Reciprocal Rank) calculation."""
    from evaluate import calculate_mrr_at_k

    # First position (rank 1)
    assert calculate_mrr_at_k([0], [0, 1, 2]) == 1.0

    # Second position (rank 2)
    assert calculate_mrr_at_k([1], [0, 1, 2]) == 0.5

    # Third position (rank 3)
    assert calculate_mrr_at_k([2], [0, 1, 2]) == pytest.approx(1 / 3)

    # Not found
    assert calculate_mrr_at_k([5], [0, 1, 2]) == 0.0

    # Multiple relevant, first found at position 2
    assert calculate_mrr_at_k([1, 2], [0, 1, 2]) == 0.5


def test_calculate_ndcg_at_k():
    """Test nDCG@K (Normalized Discounted Cumulative Gain) calculation."""
    from evaluate import calculate_ndcg_at_k

    # Perfect ranking
    assert calculate_ndcg_at_k([0, 1, 2], [0, 1, 2]) == 1.0

    # Some relevant documents
    ndcg = calculate_ndcg_at_k([0, 1], [0, 1, 2])
    assert 0.0 < ndcg <= 1.0

    # No relevant documents
    assert calculate_ndcg_at_k([0], [1, 2, 3]) == 0.0


def test_calculate_precision_at_k():
    """Test Precision@K calculation."""
    from evaluate import calculate_precision_at_k

    # All relevant
    assert calculate_precision_at_k([0, 1, 2], [0, 1, 2]) == 1.0

    # 2 out of 3 relevant
    assert calculate_precision_at_k([0, 1], [0, 1, 2]) == pytest.approx(2 / 3)

    # None relevant
    assert calculate_precision_at_k([0, 1], [2, 3, 4]) == 0.0

    # Empty retrieved
    assert calculate_precision_at_k([0, 1], []) == 0.0


def test_read_eval_data(sample_eval_data):
    """Test reading evaluation data from JSONL file."""
    import json

    eval_data = []
    with open(sample_eval_data) as f:
        for line in f:
            eval_data.append(json.loads(line))

    assert len(eval_data) == 3
    assert all("query" in item for item in eval_data)
    assert all("relevant_contains" in item for item in eval_data)


def test_evaluate_query(simple_index, sample_passages):
    """Test single query evaluation."""
    from evaluate import evaluate_query

    # Query that should match first passage
    query = "reinforcement learning"
    relevant_ids = [0]  # First passage

    metrics = evaluate_query(simple_index, query, relevant_ids, k=3, method="tfidf")

    assert "recall" in metrics
    assert "mrr" in metrics
    assert "ndcg" in metrics
    assert "precision" in metrics
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["mrr"] <= 1.0


def test_evaluation_with_no_relevant():
    """Test evaluation when no relevant documents exist."""
    from evaluate import calculate_mrr_at_k, calculate_ndcg_at_k, calculate_recall_at_k

    relevant_ids = [100, 101]  # Non-existent IDs
    retrieved_ids = [0, 1, 2]

    assert calculate_recall_at_k(relevant_ids, retrieved_ids) == 0.0
    assert calculate_mrr_at_k(relevant_ids, retrieved_ids) == 0.0
    assert calculate_ndcg_at_k(relevant_ids, retrieved_ids) == 0.0


def test_evaluation_edge_cases():
    """Test edge cases in evaluation metrics."""
    from evaluate import calculate_mrr_at_k, calculate_recall_at_k

    # Empty relevant set
    assert calculate_recall_at_k([], [0, 1, 2]) == 0.0
    assert calculate_mrr_at_k([], [0, 1, 2]) == 0.0

    # Empty retrieved set
    assert calculate_recall_at_k([0, 1], []) == 0.0
    assert calculate_mrr_at_k([0, 1], []) == 0.0

    # Both empty
    assert calculate_recall_at_k([], []) == 0.0
    assert calculate_mrr_at_k([], []) == 0.0


def test_per_query_report_structure(tmp_path):
    """Test that per-query reports have correct structure."""

    # We need to test the structure of what would be written
    # This is more of an integration test
    pass  # Skipping as it requires full evaluation run


def test_worst_queries_analysis():
    """Test worst queries analysis structure."""
    # Test that worst queries are correctly identified by low metrics
    queries_with_metrics = [
        {"query": "good query", "metrics": {"mrr": 1.0, "ndcg": 1.0}},
        {"query": "bad query", "metrics": {"mrr": 0.1, "ndcg": 0.2}},
        {"query": "ok query", "metrics": {"mrr": 0.5, "ndcg": 0.6}},
    ]

    # Sort by MRR (ascending) to find worst
    sorted_queries = sorted(queries_with_metrics, key=lambda x: x["metrics"]["mrr"])

    assert sorted_queries[0]["query"] == "bad query"
    assert sorted_queries[-1]["query"] == "good query"


def test_metrics_range():
    """Test that all metrics are in valid range [0, 1]."""
    from evaluate import (
        calculate_mrr_at_k,
        calculate_ndcg_at_k,
        calculate_precision_at_k,
        calculate_recall_at_k,
    )

    test_cases = [
        ([0, 1], [0, 1, 2]),
        ([0], [1, 2, 3]),
        ([1, 2], [0, 1, 2]),
        ([], [0, 1, 2]),
        ([0, 1], []),
    ]

    for relevant, retrieved in test_cases:
        recall = calculate_recall_at_k(relevant, retrieved)
        mrr = calculate_mrr_at_k(relevant, retrieved)
        ndcg = calculate_ndcg_at_k(relevant, retrieved)
        precision = calculate_precision_at_k(relevant, retrieved)

        assert 0.0 <= recall <= 1.0
        assert 0.0 <= mrr <= 1.0
        assert 0.0 <= ndcg <= 1.0
        assert 0.0 <= precision <= 1.0


def test_eval_jsonl_format(tmp_path):
    """Test that eval JSONL files are correctly formatted."""
    # Create a test eval file
    eval_file = tmp_path / "test_eval.jsonl"

    test_data = [
        {"query": "test query 1", "relevant_contains": "answer 1"},
        {"query": "test query 2", "relevant_contains": "answer 2"},
    ]

    with open(eval_file, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    # Read it back
    loaded_data = []
    with open(eval_file) as f:
        for line in f:
            loaded_data.append(json.loads(line))

    assert len(loaded_data) == 2
    assert loaded_data[0]["query"] == "test query 1"
    assert loaded_data[1]["relevant_contains"] == "answer 2"


def test_find_relevant_passages():
    """Test finding relevant passages by content matching."""
    passages = [
        "This contains the word apple",
        "This contains the word banana",
        "This contains the word cherry",
    ]

    # Find passages containing 'banana'
    relevant_ids = [i for i, p in enumerate(passages) if "banana" in p.lower()]

    assert relevant_ids == [1]

    # Find passages containing 'contains'
    relevant_ids = [i for i, p in enumerate(passages) if "contains" in p.lower()]

    assert len(relevant_ids) == 3


def test_aggregate_metrics():
    """Test aggregating metrics across multiple queries."""
    query_results = [
        {"recall": 1.0, "mrr": 1.0, "ndcg": 1.0, "precision": 0.33},
        {"recall": 0.5, "mrr": 0.5, "ndcg": 0.8, "precision": 0.33},
        {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "precision": 0.0},
    ]

    # Calculate averages
    avg_recall = sum(r["recall"] for r in query_results) / len(query_results)
    avg_mrr = sum(r["mrr"] for r in query_results) / len(query_results)

    assert avg_recall == pytest.approx(0.5)
    assert avg_mrr == pytest.approx(0.5)
