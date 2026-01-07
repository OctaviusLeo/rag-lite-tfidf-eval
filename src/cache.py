"""
Caching layer for RAG-Lite.

Provides file-based caching for embeddings and query results,
with optional Redis support for distributed caching.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np


class CacheManager:
    """Manager for file-based caching."""

    def __init__(self, cache_dir: str = ".cache/rag-lite", max_size_mb: int = 1000):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files
            max_size_mb: Maximum cache size in megabytes
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.embeddings_dir = self.cache_dir / "embeddings"
        self.queries_dir = self.cache_dir / "queries"
        self.embeddings_dir.mkdir(exist_ok=True)
        self.queries_dir.mkdir(exist_ok=True)

    def _get_hash(self, key: str) -> str:
        """Generate hash for cache key."""
        return hashlib.sha256(key.encode()).hexdigest()

    def _get_cache_size(self) -> float:
        """Get current cache size in MB."""
        total_size = 0
        for dirpath, _dirnames, filenames in os.walk(self.cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)

    def _cleanup_if_needed(self) -> None:
        """Remove old cache files if size exceeds limit."""
        current_size = self._get_cache_size()
        if current_size <= self.max_size_mb:
            return

        # Get all cache files with their access times
        cache_files = []
        for dirpath, _dirnames, filenames in os.walk(self.cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                atime = os.path.getatime(filepath)
                size = os.path.getsize(filepath)
                cache_files.append((filepath, atime, size))

        # Sort by access time (oldest first)
        cache_files.sort(key=lambda x: x[1])

        # Remove files until we're under the limit
        target_size = self.max_size_mb * 0.8  # Remove to 80% of limit
        for filepath, _, size in cache_files:
            if current_size <= target_size:
                break
            try:
                os.remove(filepath)
                current_size -= size / (1024 * 1024)
            except OSError:
                pass

    def get_embedding_cache_path(self, text: str, model: str) -> Path:
        """Get cache file path for text embedding."""
        key = f"{model}:{text}"
        hash_key = self._get_hash(key)
        return self.embeddings_dir / f"{hash_key}.pkl"

    def get_embedding(self, text: str, model: str) -> np.ndarray | None:
        """
        Retrieve cached embedding.

        Args:
            text: Text that was embedded
            model: Model name used for embedding

        Returns:
            Cached embedding or None if not found
        """
        cache_path = self.get_embedding_cache_path(text, model)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def set_embedding(self, text: str, model: str, embedding: np.ndarray) -> None:
        """
        Cache an embedding.

        Args:
            text: Text that was embedded
            model: Model name used for embedding
            embedding: Embedding vector to cache
        """
        cache_path = self.get_embedding_cache_path(text, model)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
            self._cleanup_if_needed()
        except Exception:
            pass

    def get_query_cache_path(self, query: str, method: str, k: int, index_hash: str) -> Path:
        """Get cache file path for query result."""
        key = f"{index_hash}:{method}:{k}:{query}"
        hash_key = self._get_hash(key)
        return self.queries_dir / f"{hash_key}.json"

    def get_query_result(
        self, query: str, method: str, k: int, index_hash: str, ttl: int = 3600
    ) -> list[tuple[str, float]] | None:
        """
        Retrieve cached query result.

        Args:
            query: Query string
            method: Retrieval method
            k: Number of results
            index_hash: Hash of the index
            ttl: Time-to-live in seconds

        Returns:
            Cached results or None if not found/expired
        """
        cache_path = self.get_query_cache_path(query, method, k, index_hash)
        if not cache_path.exists():
            return None

        try:
            # Check if expired
            if time.time() - os.path.getmtime(cache_path) > ttl:
                os.remove(cache_path)
                return None

            with open(cache_path) as f:
                data = json.load(f)
            return [(item['text'], item['score']) for item in data]
        except Exception:
            return None

    def set_query_result(
        self, query: str, method: str, k: int, index_hash: str, results: list[tuple[str, float]]
    ) -> None:
        """
        Cache query results.

        Args:
            query: Query string
            method: Retrieval method
            k: Number of results
            index_hash: Hash of the index
            results: Query results to cache
        """
        cache_path = self.get_query_cache_path(query, method, k, index_hash)
        try:
            data = [{'text': text, 'score': score} for text, score in results]
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            self._cleanup_if_needed()
        except Exception:
            pass

    def clear_all(self) -> None:
        """Clear all cache."""
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.__init__(str(self.cache_dir), self.max_size_mb)

    def clear_embeddings(self) -> None:
        """Clear embedding cache only."""
        shutil.rmtree(self.embeddings_dir, ignore_errors=True)
        self.embeddings_dir.mkdir(exist_ok=True)

    def clear_queries(self) -> None:
        """Clear query cache only."""
        shutil.rmtree(self.queries_dir, ignore_errors=True)
        self.queries_dir.mkdir(exist_ok=True)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        def count_files(directory: Path) -> int:
            return len(list(directory.glob("*"))) if directory.exists() else 0

        return {
            "total_size_mb": self._get_cache_size(),
            "max_size_mb": self.max_size_mb,
            "num_embeddings": count_files(self.embeddings_dir),
            "num_queries": count_files(self.queries_dir),
            "cache_dir": str(self.cache_dir),
        }


class RedisCache:
    """Redis-based cache for distributed environments (optional)."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, ttl: int = 3600):
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            ttl: Default time-to-live in seconds
        """
        try:
            import redis
            self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=False)
            self.ttl = ttl
            self.available = True
            # Test connection
            self.redis.ping()
        except (ImportError, Exception):
            self.available = False
            print("⚠ Redis not available, falling back to file cache")

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if not self.available:
            return None
        try:
            data = self.redis.get(key)
            return pickle.loads(data) if data else None
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache."""
        if not self.available:
            return
        try:
            data = pickle.dumps(value)
            self.redis.setex(key, ttl or self.ttl, data)
        except Exception:
            pass

    def delete(self, key: str) -> None:
        """Delete key from cache."""
        if not self.available:
            return
        with suppress(Exception):
            self.redis.delete(key)

    def clear_all(self) -> None:
        """Clear all cache."""
        if not self.available:
            return
        with suppress(Exception):
            self.redis.flushdb()


# Global cache instance
_cache_manager: CacheManager | None = None


def get_cache_manager(cache_dir: str = ".cache/rag-lite", max_size_mb: int = 1000) -> CacheManager:
    """Get or create global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(cache_dir, max_size_mb)
    return _cache_manager


if __name__ == "__main__":
    # Demo cache usage
    cache = CacheManager()
    print("Cache stats:", cache.get_stats())
