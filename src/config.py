"""
Configuration management for RAG-Lite.

Supports YAML and TOML configuration files for retrieval parameters,
model settings, caching, and system configuration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import yaml


@dataclass
class RetrievalConfig:
    """Configuration for retrieval methods."""
    default_method: Literal["tfidf", "bm25", "embeddings", "hybrid"] = "tfidf"
    default_k: int = 10
    bm25_weight: float = 0.5  # Weight for hybrid retrieval
    enable_reranking: bool = False


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32
    device: str | None = None  # None = auto-detect


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    enabled: bool = False
    chunk_size: int = 200
    overlap: int = 50


@dataclass
class CacheConfig:
    """Configuration for caching."""
    enabled: bool = True
    cache_dir: str = ".cache/rag-lite"
    embeddings_cache: bool = True
    query_cache: bool = True
    query_cache_ttl: int = 3600  # seconds
    max_cache_size_mb: int = 1000


@dataclass
class APIConfig:
    """Configuration for REST API."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    log_level: str = "info"


@dataclass
class Config:
    """Main configuration container."""
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    api: APIConfig = field(default_factory=APIConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        """Load configuration from TOML file."""
        with open(path, 'rb') as f:
            data = tomllib.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Create configuration from dictionary."""
        return cls(
            retrieval=RetrievalConfig(**data.get('retrieval', {})),
            models=ModelConfig(**data.get('models', {})),
            chunking=ChunkingConfig(**data.get('chunking', {})),
            cache=CacheConfig(**data.get('cache', {})),
            api=APIConfig(**data.get('api', {})),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        """
        Load configuration from file (auto-detect format).

        Supports .yaml, .yml, and .toml files.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        if path.suffix in ['.yaml', '.yml']:
            return cls.from_yaml(path)
        elif path.suffix == '.toml':
            return cls.from_toml(path)
        else:
            raise ValueError(f"Unsupported configuration format: {path.suffix}")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'retrieval': self.retrieval.__dict__,
            'models': self.models.__dict__,
            'chunking': self.chunking.__dict__,
            'cache': self.cache.__dict__,
            'api': self.api.__dict__,
        }


def load_config(config_path: str | Path | None = None) -> Config:
    """
    Load configuration with fallback logic.

    Priority:
    1. Provided config_path
    2. Environment variable RAG_LITE_CONFIG
    3. config.yaml in current directory
    4. config.toml in current directory
    5. Default configuration
    """
    # Check provided path
    if config_path:
        return Config.from_file(config_path)

    # Check environment variable
    env_config = os.environ.get('RAG_LITE_CONFIG')
    if env_config and os.path.exists(env_config):
        return Config.from_file(env_config)

    # Check default locations
    for default_path in ['config.yaml', 'config.yml', 'config.toml']:
        if os.path.exists(default_path):
            return Config.from_file(default_path)

    # Return default configuration
    return Config()


def create_default_config_yaml(output_path: str | Path = "config.yaml") -> None:
    """Create a default configuration file in YAML format."""
    config = Config()

    yaml_content = """# RAG-Lite Configuration

# Retrieval settings
retrieval:
  default_method: tfidf  # tfidf, bm25, embeddings, hybrid
  default_k: 10
  bm25_weight: 0.5  # For hybrid retrieval
  enable_reranking: false

# Model settings
models:
  embedder_model: sentence-transformers/all-MiniLM-L6-v2
  reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
  batch_size: 32
  device: null  # null for auto-detect, or "cpu"/"cuda"

# Document chunking
chunking:
  enabled: false
  chunk_size: 200
  overlap: 50

# Caching
cache:
  enabled: true
  cache_dir: .cache/rag-lite
  embeddings_cache: true
  query_cache: true
  query_cache_ttl: 3600  # seconds
  max_cache_size_mb: 1000

# API server settings
api:
  host: 0.0.0.0
  port: 8000
  reload: false
  workers: 1
  log_level: info
"""

    with open(output_path, 'w') as f:
        f.write(yaml_content)

    print(f"✓ Created default configuration: {output_path}")


def create_default_config_toml(output_path: str | Path = "config.toml") -> None:
    """Create a default configuration file in TOML format."""
    toml_content = """# RAG-Lite Configuration

[retrieval]
default_method = "tfidf"  # tfidf, bm25, embeddings, hybrid
default_k = 10
bm25_weight = 0.5  # For hybrid retrieval
enable_reranking = false

[models]
embedder_model = "sentence-transformers/all-MiniLM-L6-v2"
reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
batch_size = 32
device = "auto"  # auto, cpu, or cuda

[chunking]
enabled = false
chunk_size = 200
overlap = 50

[cache]
enabled = true
cache_dir = ".cache/rag-lite"
embeddings_cache = true
query_cache = true
query_cache_ttl = 3600  # seconds
max_cache_size_mb = 1000

[api]
host = "0.0.0.0"
port = 8000
reload = false
workers = 1
log_level = "info"
"""

    with open(output_path, 'w') as f:
        f.write(toml_content)

    print(f"✓ Created default configuration: {output_path}")


if __name__ == "__main__":
    # Generate default config files
    create_default_config_yaml()
    create_default_config_toml()
