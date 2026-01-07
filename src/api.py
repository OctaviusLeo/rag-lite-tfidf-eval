"""
FastAPI REST API for RAG-Lite.

Provides HTTP endpoints for building indices, querying, and system health checks.
"""
from __future__ import annotations

import os
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import psutil
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag import Index, build_index, retrieve, retrieve_hybrid
from src.io_utils import read_text


# Global state
app_state = {"index": None, "index_path": None, "metrics": {"queries": 0, "errors": 0}}


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="Query string to search for", min_length=1)
    k: int = Field(default=3, ge=1, le=100, description="Number of results to return")
    method: Literal["tfidf", "bm25", "embeddings", "hybrid"] = Field(
        default="tfidf", description="Retrieval method to use"
    )
    rerank: bool = Field(default=False, description="Apply cross-encoder reranking")
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for BM25 in hybrid mode")


class QueryResult(BaseModel):
    """Single result from a query."""
    rank: int
    text: str
    score: float


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    query: str
    method: str
    k: int
    results: list[QueryResult]
    latency_ms: float


class BuildIndexRequest(BaseModel):
    """Request model for building an index."""
    docs_path: str = Field(..., description="Path to documents file")
    output_path: str = Field(default="outputs/index.pkl", description="Path to save index")
    bm25: bool = Field(default=False, description="Enable BM25")
    embeddings: bool = Field(default=False, description="Enable dense embeddings")
    reranker: bool = Field(default=False, description="Load reranker model")
    chunking: bool = Field(default=False, description="Enable document chunking")
    chunk_size: int = Field(default=200, ge=50, le=1000, description="Chunk size in characters")
    overlap: int = Field(default=50, ge=0, le=500, description="Overlap between chunks")


class BuildIndexResponse(BaseModel):
    """Response model for build index endpoint."""
    status: str
    output_path: str
    num_passages: int
    num_chunks: int | None = None
    build_time_seconds: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    index_loaded: bool
    index_path: str | None = None


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""
    queries_total: int
    errors_total: int
    system: dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI app."""
    # Startup: Try to load default index if it exists
    default_index = "outputs/index.pkl"
    if os.path.exists(default_index):
        try:
            with open(default_index, "rb") as f:
                app_state["index"] = pickle.load(f)
                app_state["index_path"] = default_index
            print(f"✓ Loaded index from {default_index}")
        except Exception as e:
            print(f"⚠ Could not load default index: {e}")

    yield

    # Shutdown: Clean up if needed
    app_state["index"] = None


# Create FastAPI app
app = FastAPI(
    title="RAG-Lite API",
    description="Production-grade retrieval system with multiple methods",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG-Lite API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        index_loaded=app_state["index"] is not None,
        index_path=app_state["index_path"],
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["General"])
async def get_metrics():
    """Get API metrics and system information."""
    process = psutil.Process()
    memory_info = process.memory_info()

    return MetricsResponse(
        queries_total=app_state["metrics"]["queries"],
        errors_total=app_state["metrics"]["errors"],
        system={
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_mb": memory_info.rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
        },
    )


@app.post("/query", response_model=QueryResponse, tags=["Retrieval"])
async def query_index(request: QueryRequest):
    """
    Query the loaded index.

    Returns the top-k most relevant passages for the given query.
    """
    if app_state["index"] is None:
        raise HTTPException(status_code=503, detail="No index loaded. Build or load an index first.")

    start_time = time.time()

    try:
        index: Index = app_state["index"]

        # Perform retrieval using retrieve_hybrid which supports all methods
        idx_score_text = retrieve_hybrid(
            index,
            request.query,
            k=request.k,
            method=request.method,
            rerank=request.rerank,
        )

        # Convert to (text, score) format
        results = [(text, score) for idx, score, text in idx_score_text]

        latency_ms = (time.time() - start_time) * 1000
        app_state["metrics"]["queries"] += 1

        return QueryResponse(
            query=request.query,
            method=request.method,
            k=request.k,
            results=[
                QueryResult(rank=i + 1, text=text, score=score)
                for i, (text, score) in enumerate(results)
            ],
            latency_ms=latency_ms,
        )

    except Exception as e:
        app_state["metrics"]["errors"] += 1
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/build-index", response_model=BuildIndexResponse, tags=["Index Management"])
async def build_new_index(request: BuildIndexRequest, background_tasks: BackgroundTasks):
    """
    Build a new index from documents.

    This endpoint builds the index synchronously and returns when complete.
    For large document sets, consider increasing timeout settings.
    """
    if not os.path.exists(request.docs_path):
        raise HTTPException(status_code=404, detail=f"Documents file not found: {request.docs_path}")

    try:
        start_time = time.time()

        # Read documents
        passages = read_text(request.docs_path)

        # Build index
        index = build_index(
            passages,
            use_bm25=request.bm25,
            use_embeddings=request.embeddings,
            use_reranker=request.reranker,
            use_chunking=request.chunking,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
        )

        # Save index
        os.makedirs(os.path.dirname(request.output_path) or ".", exist_ok=True)
        with open(request.output_path, "wb") as f:
            pickle.dump(index, f)

        # Update app state
        app_state["index"] = index
        app_state["index_path"] = request.output_path

        build_time = time.time() - start_time

        return BuildIndexResponse(
            status="success",
            output_path=request.output_path,
            num_passages=len(passages),
            num_chunks=len(index.chunks) if index.chunks else None,
            build_time_seconds=build_time,
        )

    except Exception as e:
        app_state["metrics"]["errors"] += 1
        raise HTTPException(status_code=500, detail=f"Index build failed: {str(e)}")


@app.post("/load-index", tags=["Index Management"])
async def load_existing_index(index_path: str):
    """
    Load an existing index from disk.

    Args:
        index_path: Path to the index pickle file
    """
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail=f"Index file not found: {index_path}")

    try:
        with open(index_path, "rb") as f:
            index = pickle.load(f)

        app_state["index"] = index
        app_state["index_path"] = index_path

        return {
            "status": "success",
            "message": f"Loaded index from {index_path}",
            "num_passages": len(index.passages),
        }

    except Exception as e:
        app_state["metrics"]["errors"] += 1
        raise HTTPException(status_code=500, detail=f"Failed to load index: {str(e)}")


def main():
    """Run the API server."""
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
