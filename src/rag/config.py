"""
RAG Configuration Models.

Pydantic models for RAG configuration with validation and defaults.
"""

from pydantic import BaseModel, Field, field_validator


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model: str = Field(
        default="BAAI/bge-m3",
        description="Embedding model name (BGE-M3 for multilingual, 8192 token context)",
    )
    device: str = Field(
        default="cuda",
        pattern="^(cuda|cpu)$",
        description="Device to use for embeddings (cuda or cpu)",
    )
    batch_size: int = Field(
        default=32, ge=1, le=128, description="Batch size for embedding generation"
    )
    dimensions: int = Field(
        default=1024, description="Embedding dimensions (BGE-M3: 1024)"
    )


class ChunkingConfig(BaseModel):
    """Document chunking configuration."""

    chunk_size: int = Field(
        default=512, ge=128, le=2048, description="Chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=50, ge=0, le=512, description="Chunk overlap in tokens"
    )
    separators: list[str] = Field(
        default=["\n\n", "\n", ". ", " ", ""],
        description="Text separators for recursive splitting",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class RerankerConfig(BaseModel):
    """Reranking configuration."""

    enable: bool = Field(
        default=True, description="Enable reranking with cross-encoder"
    )
    model: str = Field(
        default="BAAI/bge-reranker-v2-base", description="Reranker model name"
    )
    top_k: int = Field(
        default=30,
        ge=5,
        le=100,
        description="Number of documents to recall before reranking",
    )
    final_k: int = Field(
        default=5, ge=1, le=20, description="Final number of documents after reranking"
    )
    batch_size: int = Field(
        default=16, ge=1, le=64, description="Batch size for reranking"
    )

    @field_validator("final_k")
    @classmethod
    def validate_final_k(cls, v: int, info) -> int:
        """Ensure final_k is less than top_k."""
        top_k = info.data.get("top_k", 30)
        if v >= top_k:
            raise ValueError("final_k must be less than top_k")
        return v


class SearchConfig(BaseModel):
    """Hybrid search configuration."""

    default_top_k: int = Field(
        default=5, ge=1, le=20, description="Default number of search results"
    )
    hybrid_search: bool = Field(
        default=True, description="Enable hybrid search (dense + sparse)"
    )
    dense_weight: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Weight for dense vector search"
    )
    sparse_weight: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Weight for sparse (BM25) search"
    )
    rrf_k: int = Field(default=60, ge=1, le=100, description="RRF constant k")

    @field_validator("dense_weight", "sparse_weight")
    @classmethod
    def validate_weights(cls, v: float, info) -> float:
        """Ensure weights sum to approximately 1.0."""
        # This is a soft validation - actual normalization happens in search
        return v


class RAGConfig(BaseModel):
    """Complete RAG configuration."""

    enable: bool = Field(default=False, description="Enable RAG functionality")
    qdrant_url: str = Field(default="http://127.0.0.1:6333", description="Qdrant URL")
    qdrant_timeout: int = Field(
        default=10, ge=1, le=60, description="Qdrant timeout in seconds"
    )
    prefer_grpc: bool = Field(
        default=True, description="Prefer gRPC for Qdrant (better performance)"
    )

    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig, description="Embedding model configuration"
    )

    chunking: ChunkingConfig = Field(
        default_factory=ChunkingConfig, description="Document chunking configuration"
    )

    search: SearchConfig = Field(
        default_factory=SearchConfig, description="Search configuration"
    )

    reranker: RerankerConfig = Field(
        default_factory=RerankerConfig, description="Reranking configuration"
    )

    max_document_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024,
        le=100 * 1024 * 1024,  # 100MB
        description="Maximum document size in bytes",
    )

    supported_formats: list[str] = Field(
        default=["txt", "md", "pdf"], description="Supported document formats"
    )

    @field_validator("search")
    @classmethod
    def validate_search_weights(cls, v: SearchConfig) -> SearchConfig:
        """Normalize search weights to sum to 1.0."""
        total = v.dense_weight + v.sparse_weight
        if total > 0:
            v.dense_weight = v.dense_weight / total
            v.sparse_weight = v.sparse_weight / total
        return v
