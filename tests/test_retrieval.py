"""Tests for retrieval functionality."""

import pytest


def test_build_simple_index(sample_passages):
    """Test building a simple TF-IDF index."""
    from rag import build_index

    index = build_index(
        sample_passages,
        use_chunking=False,
        use_bm25=False,
        use_embeddings=False,
        use_reranker=False,
    )

    assert index is not None
    assert hasattr(index, "passages")
    assert hasattr(index, "vectorizer")
    assert len(index.passages) == len(sample_passages)


def test_tfidf_retrieval(simple_index):
    """Test TF-IDF retrieval returns relevant results."""
    from rag import retrieve

    results = retrieve(simple_index, "What is reinforcement learning?", k=3)

    assert len(results) <= 3
    # Results are tuples (doc_id, score, passage)
    assert all(len(r) == 3 for r in results)
    # First result should contain "Reinforcement learning"
    assert "Reinforcement learning" in results[0][2] or "reinforcement" in results[0][2].lower()


def test_bm25_retrieval(hybrid_index):
    """Test BM25 retrieval."""
    from rag import retrieve_hybrid

    results = retrieve_hybrid(hybrid_index, "reinforcement learning agent", k=3, method="bm25")

    assert len(results) <= 3
    # Results are tuples (doc_id, score, passage)
    assert all(len(r) == 3 for r in results)


@pytest.mark.slow
def test_embeddings_retrieval(hybrid_index):
    """Test embeddings-based retrieval."""
    from rag import retrieve_hybrid

    results = retrieve_hybrid(hybrid_index, "machine learning", k=3, method="embeddings")

    assert len(results) <= 3
    # Results are tuples (doc_id, score, passage)
    assert all(len(r) == 3 for r in results)


@pytest.mark.slow
def test_hybrid_retrieval(hybrid_index):
    """Test hybrid retrieval combining multiple methods."""
    from rag import retrieve_hybrid

    results = retrieve_hybrid(
        hybrid_index,
        "deep learning neural networks",
        k=3,
        method="hybrid",
        tfidf_weight=0.4,
        bm25_weight=0.3,
        embedding_weight=0.3,
    )

    assert len(results) <= 3
    # Results are tuples (doc_id, score, passage)
    assert all(len(r) == 3 for r in results)


def test_chunking(sample_passages):
    """Test chunking functionality."""
    from rag import build_index

    chunked_index = build_index(sample_passages, use_chunking=True, chunk_size=100, overlap=20)

    assert chunked_index.chunks is not None
    # Should have more chunks than original passages
    assert len(chunked_index.chunks) >= len(sample_passages)

    # Test retrieval on chunked index
    from rag import retrieve

    results = retrieve(chunked_index, "What is TF-IDF?", k=3)
    assert len(results) > 0


def test_grounded_retrieval(chunked_index):
    """Test grounded retrieval with citations."""
    from rag import retrieve_grounded

    results = retrieve_grounded(chunked_index, "reinforcement learning", k=3, method="tfidf")

    assert len(results) <= 3
    for result in results:
        assert hasattr(result, "chunk_id")
        assert hasattr(result, "source_doc_id")  # Changed from doc_id
        assert hasattr(result, "score")
        assert hasattr(result, "text")
        assert hasattr(result, "char_range")  # Changed from start_char/end_char


def test_retrieve_empty_query(simple_index):
    """Test retrieval with empty query."""
    from rag import retrieve

    results = retrieve(simple_index, "", k=3)

    # Should still return some results (all with zero scores or all equally ranked)
    assert isinstance(results, list)


def test_retrieve_k_parameter(simple_index):
    """Test that k parameter is respected."""
    from rag import retrieve

    for k in [1, 3, 5]:
        results = retrieve(simple_index, "learning", k=k)
        assert len(results) <= k


def test_chunk_text():
    """Test chunk_text function."""
    from rag import chunk_text

    text = "This is sentence one. This is sentence two. This is sentence three. End."
    chunks = chunk_text(text, chunk_size=30, overlap=10)

    assert len(chunks) > 1
    # Check that chunks have text attribute
    for chunk in chunks:
        assert hasattr(chunk, "text")
        assert len(chunk.text) > 0


def test_score_ordering(simple_index):
    """Test that results are ordered by score descending."""
    from rag import retrieve

    results = retrieve(simple_index, "machine learning AI", k=5)

    # Check scores are in descending order
    # Results are tuples (doc_id, score, passage)
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_index_persistence(tmp_path, sample_passages):
    """Test saving and loading index."""
    import pickle

    from rag import build_index

    # Build index
    index = build_index(sample_passages)

    # Save index
    index_path = tmp_path / "test_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump(index, f)

    # Load index
    with open(index_path, "rb") as f:
        loaded_index = pickle.load(f)

    assert loaded_index is not None
    assert len(loaded_index.passages) == len(sample_passages)


@pytest.mark.slow
def test_reranker(hybrid_index):
    """Test cross-encoder reranker is loaded."""
    # Test that the reranker is loaded in hybrid index
    assert hasattr(hybrid_index, "reranker")
    assert hybrid_index.reranker is not None
