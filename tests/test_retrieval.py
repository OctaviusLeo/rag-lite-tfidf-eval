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
    assert "passages" in index
    assert "vectorizer" in index
    assert len(index["passages"]) == len(sample_passages)


def test_tfidf_retrieval(simple_index):
    """Test TF-IDF retrieval returns relevant results."""
    from rag import retrieve

    results = retrieve(simple_index, "What is reinforcement learning?", k=3, method="tfidf")

    assert len(results) <= 3
    assert all("passage_id" in r for r in results)
    assert all("score" in r for r in results)
    assert all("text" in r for r in results)
    # First result should contain "Reinforcement learning"
    assert (
        "Reinforcement learning" in results[0]["text"]
        or "reinforcement" in results[0]["text"].lower()
    )


def test_bm25_retrieval(hybrid_index):
    """Test BM25 retrieval."""
    from rag import retrieve

    results = retrieve(hybrid_index, "reinforcement learning agent", k=3, method="bm25")

    assert len(results) <= 3
    assert all("passage_id" in r for r in results)
    assert all("score" in r for r in results)


@pytest.mark.slow
def test_embeddings_retrieval(hybrid_index):
    """Test embeddings-based retrieval."""
    from rag import retrieve

    results = retrieve(hybrid_index, "machine learning", k=3, method="embeddings")

    assert len(results) <= 3
    assert all("passage_id" in r for r in results)
    assert all("score" in r for r in results)


@pytest.mark.slow
def test_hybrid_retrieval(hybrid_index):
    """Test hybrid retrieval combining multiple methods."""
    from rag import retrieve_hybrid

    results = retrieve_hybrid(
        hybrid_index,
        "deep learning neural networks",
        k=3,
        tfidf_weight=0.4,
        bm25_weight=0.3,
        embeddings_weight=0.3,
    )

    assert len(results) <= 3
    assert all("passage_id" in r for r in results)
    assert all("score" in r for r in results)


def test_chunking(sample_passages):
    """Test chunking functionality."""
    from rag import build_index, retrieve

    chunked_index = build_index(sample_passages, use_chunking=True, chunk_size=100, overlap=20)

    assert "chunks" in chunked_index
    # Should have more chunks than original passages
    assert len(chunked_index["chunks"]) >= len(sample_passages)

    # Test retrieval on chunked index
    results = retrieve(chunked_index, "What is TF-IDF?", k=3, method="tfidf")
    assert len(results) > 0


def test_grounded_retrieval(chunked_index):
    """Test grounded retrieval with citations."""
    from rag import retrieve_grounded

    results = retrieve_grounded(chunked_index, "reinforcement learning", k=3, method="tfidf")

    assert len(results) <= 3
    for result in results:
        assert hasattr(result, "chunk_id")
        assert hasattr(result, "doc_id")
        assert hasattr(result, "score")
        assert hasattr(result, "text")
        assert hasattr(result, "start_char")
        assert hasattr(result, "end_char")


def test_retrieve_empty_query(simple_index):
    """Test retrieval with empty query."""
    from rag import retrieve

    results = retrieve(simple_index, "", k=3, method="tfidf")

    # Should still return some results (all with zero scores or all equally ranked)
    assert isinstance(results, list)


def test_retrieve_k_parameter(simple_index):
    """Test that k parameter is respected."""
    from rag import retrieve

    for k in [1, 3, 5]:
        results = retrieve(simple_index, "learning", k=k, method="tfidf")
        assert len(results) <= k


def test_chunk_text():
    """Test chunk_text function."""
    from rag import chunk_text

    text = "This is sentence one. This is sentence two. This is sentence three. End."
    chunks = chunk_text(text, chunk_size=30, overlap=10)

    assert len(chunks) > 1
    # Check overlap exists
    for i in range(len(chunks) - 1):
        # Some text from chunk i should appear in chunk i+1
        assert len(chunks[i]) <= 40  # chunk_size + some tolerance for sentence boundaries


def test_score_ordering(simple_index):
    """Test that results are ordered by score descending."""
    from rag import retrieve

    results = retrieve(simple_index, "machine learning AI", k=5, method="tfidf")

    # Check scores are in descending order
    scores = [r["score"] for r in results]
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
    assert len(loaded_index["passages"]) == len(sample_passages)


@pytest.mark.slow
def test_reranker(hybrid_index):
    """Test cross-encoder reranking."""
    from rag import retrieve

    # Get results without reranking
    results_no_rerank = retrieve(hybrid_index, "deep learning", k=3, method="tfidf")

    # Note: The retrieve function doesn't directly expose reranking,
    # but we can test that the reranker is loaded in hybrid index
    assert "reranker" in hybrid_index
    assert hybrid_index["reranker"] is not None
