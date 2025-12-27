"""Pytest configuration and fixtures for RAG-Lite tests."""

import json
import sys
from pathlib import Path

import pytest


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_passages():
    """Sample passages for testing."""
    return [
        "Reinforcement learning (RL) is a learning paradigm where an agent interacts with an environment, receives rewards, and learns a policy to maximize expected return.",
        "In robotics, a common control loop is sense -> perceive -> plan/control -> act. Perception can use cameras, lidar, or other sensors to estimate state.",
        "TF-IDF is a retrieval technique that weights terms by frequency in a document and rarity across the corpus. It is often used as a baseline for information retrieval.",
        "Deep learning uses neural networks with multiple layers to learn hierarchical representations of data. Common architectures include CNNs and Transformers.",
        "Natural language processing (NLP) involves computational methods for analyzing and generating human language. Modern NLP heavily relies on transformer models.",
    ]


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        "What is reinforcement learning?",
        "How do robots perceive their environment?",
        "What is TF-IDF used for?",
    ]


@pytest.fixture
def sample_eval_data(tmp_path):
    """Create sample evaluation data file."""
    eval_data = [
        {
            "query": "What is reinforcement learning?",
            "relevant_contains": "Reinforcement learning",
        },
        {"query": "How do robots perceive?", "relevant_contains": "Perception"},
        {"query": "What is TF-IDF?", "relevant_contains": "TF-IDF"},
    ]

    eval_file = tmp_path / "eval.jsonl"
    with open(eval_file, "w") as f:
        for item in eval_data:
            f.write(json.dumps(item) + "\n")

    return eval_file


@pytest.fixture
def sample_corpus_file(tmp_path, sample_passages):
    """Create sample corpus file."""
    corpus_file = tmp_path / "docs.txt"
    with open(corpus_file, "w") as f:
        f.write("\n\n".join(sample_passages))
    return corpus_file


@pytest.fixture
def simple_index(sample_passages):
    """Build a simple TF-IDF index for testing."""
    from rag import build_index

    return build_index(
        sample_passages,
        use_chunking=False,
        use_bm25=False,
        use_embeddings=False,
        use_reranker=False,
    )


@pytest.fixture
def hybrid_index(sample_passages):
    """Build a hybrid index with all methods."""
    from rag import build_index

    return build_index(
        sample_passages,
        use_chunking=False,
        use_bm25=True,
        use_embeddings=True,
        use_reranker=True,
    )


@pytest.fixture
def chunked_index(sample_passages):
    """Build a chunked index for testing."""
    from rag import build_index

    return build_index(
        sample_passages,
        use_chunking=True,
        chunk_size=100,
        overlap=20,
        use_bm25=False,
        use_embeddings=False,
        use_reranker=False,
    )
