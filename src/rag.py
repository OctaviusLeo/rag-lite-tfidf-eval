# rag.py
# This file contains the implementation of the Retrieval-Augmented Generation (RAG) model.
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal
import warnings

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


@dataclass
class Index:
    vectorizer: TfidfVectorizer
    passages: List[str]
    X: np.ndarray  # TF-IDF matrix
    bm25: Optional[BM25Okapi] = None
    embedder: Optional[SentenceTransformer] = None
    passage_embeddings: Optional[np.ndarray] = None
    reranker: Optional[CrossEncoder] = None


def split_passages(corpus_text: str) -> List[str]:
    # Split on blank lines
    raw = [p.strip() for p in corpus_text.split("\n\n") if p.strip()]
    return raw


def build_index(
    corpus_text: str,
    use_bm25: bool = False,
    use_embeddings: bool = False,
    use_reranker: bool = False
) -> Index:
    """Build a retrieval index with optional hybrid methods.
    
    Args:
        corpus_text: Raw corpus text with passages separated by blank lines
        use_bm25: If True, build BM25 index for hybrid retrieval
        use_embeddings: If True, build dense embedding index
        use_reranker: If True, load a cross-encoder reranker
    
    Returns:
        Index object with requested components
    """
    passages = split_passages(corpus_text)
    
    # TF-IDF (always included as baseline)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    X = vec.fit_transform(passages)
    
    # Optional: BM25
    bm25 = None
    if use_bm25:
        tokenized = [p.lower().split() for p in passages]
        bm25 = BM25Okapi(tokenized)
    
    # Optional: Dense embeddings
    embedder = None
    passage_embeddings = None
    if use_embeddings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            passage_embeddings = embedder.encode(passages, convert_to_numpy=True, show_progress_bar=False)
    
    # Optional: Reranker
    reranker = None
    if use_reranker:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    return Index(
        vectorizer=vec,
        passages=passages,
        X=X,
        bm25=bm25,
        embedder=embedder,
        passage_embeddings=passage_embeddings,
        reranker=reranker
    )


def retrieve(index: Index, query: str, k: int = 3) -> List[Tuple[int, float, str]]:
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
    rerank_top_k: int = 20
) -> List[Tuple[int, float, str]]:
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
        query_embedding = index.embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)
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
        if index.embedder is not None and index.passage_embeddings is not None and embedding_weight > 0:
            query_embedding = index.embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)
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
        reranked = [(candidates[i][0], float(rerank_scores[i]), candidates[i][2]) 
                    for i in range(len(candidates))]
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked[:k]
    
    return candidates[:k]
