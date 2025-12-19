from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Index:
    vectorizer: TfidfVectorizer
    passages: List[str]
    X: np.ndarray  # TF-IDF matrix


def split_passages(corpus_text: str) -> List[str]:
    # Split on blank lines
    raw = [p.strip() for p in corpus_text.split("\n\n") if p.strip()]
    return raw


def build_index(corpus_text: str) -> Index:
    passages = split_passages(corpus_text)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    X = vec.fit_transform(passages)
    return Index(vectorizer=vec, passages=passages, X=X)


def retrieve(index: Index, query: str, k: int = 3) -> List[Tuple[int, float, str]]:
    q = index.vectorizer.transform([query])
    sims = cosine_similarity(q, index.X).ravel()
    top = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i]), index.passages[int(i)]) for i in top]
