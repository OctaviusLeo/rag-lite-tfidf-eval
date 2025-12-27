# rag.py
# Core retrieval implementation for multi-method information retrieval.
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    """Text chunk with metadata for citation tracking and source attribution."""

    chunk_id: str  # Stable identifier (e.g., "doc_0_chunk_0")
    text: str
    start_char: int
    end_char: int
    source_doc_id: int
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_citation(self) -> str:
        """Generate a citation string for this chunk."""
        return f"[{self.chunk_id}]"

    def get_snippet(self, max_length: int = 150) -> str:
        """Get a truncated snippet of the text."""
        if len(self.text) <= max_length:
            return self.text
        return self.text[:max_length] + "..."


@dataclass
class Index:
    vectorizer: TfidfVectorizer
    passages: list[str]
    X: np.ndarray  # TF-IDF matrix
    chunks: list[Chunk] | None = None  # Chunk metadata for grounding
    bm25: BM25Okapi | None = None
    embedder: SentenceTransformer | None = None
    passage_embeddings: np.ndarray | None = None
    reranker: CrossEncoder | None = None


def split_passages(corpus_text: str) -> list[str]:
    """Split on blank lines (legacy method)."""
    raw = [p.strip() for p in corpus_text.split("\n\n") if p.strip()]
    return raw


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50, doc_id: int = 0) -> list[Chunk]:
    """Chunk text with sliding window and metadata.

    Args:
        text: Text to chunk
        chunk_size: Target size in characters
        overlap: Overlap between chunks in characters
        doc_id: Source document ID

    Returns:
        List of Chunk objects with metadata
    """
    if not text.strip():
        return []

    chunks = []
    text_length = len(text)
    start = 0
    chunk_index = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Try to break at sentence boundary if not at the end
        if end < text_length:
            # Look for sentence endings within the last 20% of chunk
            search_start = max(start, end - int(chunk_size * 0.2))
            sentence_end = max(
                text.rfind(". ", search_start, end),
                text.rfind("! ", search_start, end),
                text.rfind("? ", search_start, end),
                text.rfind("\n", search_start, end),
            )
            if sentence_end > start:
                end = sentence_end + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk = Chunk(
                chunk_id=f"doc_{doc_id}_chunk_{chunk_index}",
                text=chunk_text,
                start_char=start,
                end_char=end,
                source_doc_id=doc_id,
                chunk_index=chunk_index,
                metadata={"length": len(chunk_text)},
            )
            chunks.append(chunk)
            chunk_index += 1

        # Move start forward, accounting for overlap
        start = end - overlap if end < text_length else text_length

        # Prevent infinite loop
        if start <= chunks[-1].start_char if chunks else True:
            start = end

    return chunks


def create_chunks_from_passages(
    passages: list[str], chunk_size: int = 200, overlap: int = 50
) -> list[Chunk]:
    """Create chunks from a list of passages.

    Args:
        passages: List of passage texts
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters

    Returns:
        List of all chunks with proper IDs and metadata
    """
    all_chunks = []
    for doc_id, passage in enumerate(passages):
        chunks = chunk_text(passage, chunk_size=chunk_size, overlap=overlap, doc_id=doc_id)
        all_chunks.extend(chunks)
    return all_chunks


def build_index(
    corpus_text: str,
    use_bm25: bool = False,
    use_embeddings: bool = False,
    use_reranker: bool = False,
    use_chunking: bool = False,
    chunk_size: int = 200,
    overlap: int = 50,
) -> Index:
    """Build a retrieval index with optional hybrid methods and chunking.

    Args:
        corpus_text: Raw corpus text with passages separated by blank lines
        use_bm25: If True, build BM25 index for hybrid retrieval
        use_embeddings: If True, build dense embedding index
        use_reranker: If True, load a cross-encoder reranker
        use_chunking: If True, chunk passages with overlap for citation grounding
        chunk_size: Target chunk size in characters (if use_chunking=True)
        overlap: Overlap between chunks in characters (if use_chunking=True)

    Returns:
        Index object with requested components
    """
    passages = split_passages(corpus_text)

    # Optionally create chunks for grounded retrieval
    chunks = None
    index_texts = passages  # Default: use full passages

    if use_chunking:
        chunks = create_chunks_from_passages(passages, chunk_size=chunk_size, overlap=overlap)
        index_texts = [chunk.text for chunk in chunks]
        print(
            f"  Chunking: {len(passages)} passages → {len(chunks)} chunks "
            f"(size={chunk_size}, overlap={overlap})"
        )

    # TF-IDF (always included as baseline)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    X = vec.fit_transform(index_texts)

    # Optional: BM25
    bm25 = None
    if use_bm25:
        tokenized = [p.lower().split() for p in index_texts]
        bm25 = BM25Okapi(tokenized)

    # Optional: Dense embeddings
    embedder = None
    passage_embeddings = None
    if use_embeddings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            passage_embeddings = embedder.encode(
                index_texts, convert_to_numpy=True, show_progress_bar=False
            )

    # Optional: Reranker
    reranker = None
    if use_reranker:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return Index(
        vectorizer=vec,
        passages=index_texts,  # Now contains chunks if chunking is enabled
        X=X,
        chunks=chunks,
        bm25=bm25,
        embedder=embedder,
        passage_embeddings=passage_embeddings,
        reranker=reranker,
    )


def retrieve(index: Index, query: str, k: int = 3) -> list[tuple[int, float, str]]:
    """Simple TF-IDF retrieval (backward compatible)."""
    q = index.vectorizer.transform([query])
    sims = cosine_similarity(q, index.X).ravel()
    top = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i]), index.passages[int(i)]) for i in top]


def retrieve_hybrid(
    index: Index,
    query: str,
    k: int = 3,
    method: Literal["tfidf", "bm25", "embeddings", "hybrid"] = "tfidf",
    tfidf_weight: float = 0.5,
    bm25_weight: float = 0.3,
    embedding_weight: float = 0.2,
    rerank: bool = False,
    rerank_top_k: int = 20,
) -> list[tuple[int, float, str]]:
    """Advanced hybrid retrieval with multiple methods.

    Args:
        index: The retrieval index
        query: Query string
        k: Number of results to return
        method: Retrieval method ("tfidf", "bm25", "embeddings", "hybrid")
        tfidf_weight: Weight for TF-IDF scores in hybrid mode
        bm25_weight: Weight for BM25 scores in hybrid mode
        embedding_weight: Weight for embedding scores in hybrid mode
        rerank: If True, apply reranking to top results
        rerank_top_k: Number of candidates to retrieve before reranking

    Returns:
        List of (doc_id, score, passage) tuples
    """
    num_docs = len(index.passages)

    if method == "tfidf":
        # TF-IDF only
        q = index.vectorizer.transform([query])
        scores = cosine_similarity(q, index.X).ravel()

    elif method == "bm25":
        # BM25 only
        if index.bm25 is None:
            raise ValueError("BM25 not available in this index")
        tokenized_query = query.lower().split()
        scores = np.array(index.bm25.get_scores(tokenized_query))
        # Normalize BM25 scores to [0, 1]
        max_score = scores.max() if scores.max() > 0 else 1.0
        scores = scores / max_score

    elif method == "embeddings":
        # Dense embeddings only
        if index.embedder is None or index.passage_embeddings is None:
            raise ValueError("Embeddings not available in this index")
        query_embedding = index.embedder.encode(
            [query], convert_to_numpy=True, show_progress_bar=False
        )
        scores = cosine_similarity(query_embedding, index.passage_embeddings).ravel()

    elif method == "hybrid":
        # Hybrid: weighted combination of multiple methods
        scores = np.zeros(num_docs)

        # TF-IDF component
        q = index.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(q, index.X).ravel()
        scores += tfidf_weight * tfidf_scores

        # BM25 component (if available)
        if index.bm25 is not None and bm25_weight > 0:
            tokenized_query = query.lower().split()
            bm25_scores = np.array(index.bm25.get_scores(tokenized_query))
            # Normalize BM25 scores
            max_score = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
            bm25_scores = bm25_scores / max_score
            scores += bm25_weight * bm25_scores

        # Embedding component (if available)
        if (
            index.embedder is not None
            and index.passage_embeddings is not None
            and embedding_weight > 0
        ):
            query_embedding = index.embedder.encode(
                [query], convert_to_numpy=True, show_progress_bar=False
            )
            emb_scores = cosine_similarity(query_embedding, index.passage_embeddings).ravel()
            scores += embedding_weight * emb_scores

    else:
        raise ValueError(f"Unknown method: {method}")

    # Get top candidates
    retrieve_k = rerank_top_k if (rerank and index.reranker is not None) else k
    retrieve_k = min(retrieve_k, num_docs)
    top_indices = np.argsort(-scores)[:retrieve_k]

    candidates = [(int(i), float(scores[i]), index.passages[int(i)]) for i in top_indices]

    # Optional reranking
    if rerank and index.reranker is not None and len(candidates) > 0:
        # Prepare pairs for cross-encoder
        pairs = [[query, passage] for _, _, passage in candidates]

        # Get reranking scores
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rerank_scores = index.reranker.predict(pairs, show_progress_bar=False)

        # Re-sort by reranking scores
        reranked = [
            (candidates[i][0], float(rerank_scores[i]), candidates[i][2])
            for i in range(len(candidates))
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked[:k]

    return candidates[:k]


@dataclass
class GroundedResult:
    """A retrieval result with citation and grounding information."""

    rank: int
    score: float
    text: str
    citation: str
    chunk_id: str | None = None
    snippet: str | None = None
    source_doc_id: int | None = None
    char_range: tuple[int, int] | None = None

    def format_result(self, include_citation: bool = True, snippet_length: int = 150) -> str:
        """Format the result for display."""
        lines = []
        if include_citation:
            lines.append(f"Citation: {self.citation}")
        lines.append(f"Score: {self.score:.3f}")
        if self.source_doc_id is not None:
            lines.append(f"Source: Document {self.source_doc_id}")
        if self.char_range:
            lines.append(f"Position: chars {self.char_range[0]}-{self.char_range[1]}")

        # Show snippet or full text
        display_text = self.snippet if self.snippet else self.text
        lines.append(f"Text: {display_text}")

        return "\n".join(lines)


def retrieve_grounded(
    index: Index,
    query: str,
    k: int = 3,
    method: Literal["tfidf", "bm25", "embeddings", "hybrid"] = "tfidf",
    rerank: bool = False,
    snippet_length: int = 150,
    **kwargs,
) -> list[GroundedResult]:
    """Retrieve with full citation and grounding information.

    Args:
        index: The retrieval index
        query: Query string
        k: Number of results to return
        method: Retrieval method
        rerank: If True, apply reranking
        snippet_length: Max length for snippets
        **kwargs: Additional arguments for retrieve_hybrid

    Returns:
        List of GroundedResult objects with citations and metadata
    """
    # Use hybrid retrieval
    raw_results = retrieve_hybrid(index, query, k=k, method=method, rerank=rerank, **kwargs)

    grounded_results = []
    for rank, (doc_id, score, text) in enumerate(raw_results, start=1):
        # Check if we have chunk metadata
        if index.chunks and doc_id < len(index.chunks):
            chunk = index.chunks[doc_id]
            result = GroundedResult(
                rank=rank,
                score=score,
                text=text,
                citation=chunk.get_citation(),
                chunk_id=chunk.chunk_id,
                snippet=chunk.get_snippet(snippet_length),
                source_doc_id=chunk.source_doc_id,
                char_range=(chunk.start_char, chunk.end_char),
            )
        else:
            # Fallback for non-chunked index
            result = GroundedResult(
                rank=rank,
                score=score,
                text=text,
                citation=f"[passage_{doc_id}]",
                chunk_id=None,
                snippet=text[:snippet_length] + "..." if len(text) > snippet_length else text,
                source_doc_id=doc_id,
                char_range=None,
            )

        grounded_results.append(result)

    return grounded_results
