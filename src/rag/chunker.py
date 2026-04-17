"""
Document Chunker for RAG.

Implements recursive character text splitting with:
- Configurable chunk size and overlap
- Semantic boundary preservation
- Metadata attachment
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from uuid import uuid4

from .config import ChunkingConfig

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A chunk of a document with metadata."""

    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    start_pos: int = 0
    end_pos: int = 0


class DocumentChunker:
    """
    Document chunker using recursive character splitting.

    Splits text hierarchically using separators to preserve semantic boundaries.
    """

    def __init__(self, config: ChunkingConfig):
        """
        Initialize document chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config

    def chunk_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Split text into chunks.

        Args:
            text: Input text
            metadata: Optional metadata to attach to chunks

        Returns:
            List of document chunks
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to chunker")
            return []

        chunks = []
        chunk_index = 0

        # Split text recursively using separators
        text_splits = self._recursive_split(text, self.config.separators)

        current_chunk = ""
        current_pos = 0

        for split in text_splits:
            # Check if adding this split would exceed chunk size
            if len(current_chunk) + len(split) > self.config.chunk_size:
                # Save current chunk if not empty
                if current_chunk:
                    chunk = DocumentChunk(
                        content=current_chunk.strip(),
                        metadata=metadata or {},
                        chunk_index=chunk_index,
                        start_pos=current_pos - len(current_chunk),
                        end_pos=current_pos,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + split
                    current_pos += len(split)
                else:
                    # Split is too large, force split
                    if len(split) > self.config.chunk_size:
                        sub_chunks = self._force_split(split)
                        for sub_chunk in sub_chunks:
                            chunk = DocumentChunk(
                                content=sub_chunk.strip(),
                                metadata=metadata or {},
                                chunk_index=chunk_index,
                                start_pos=current_pos,
                                end_pos=current_pos + len(sub_chunk),
                            )
                            chunks.append(chunk)
                            chunk_index += 1
                            current_pos += len(sub_chunk)
                    else:
                        current_chunk = split
                        current_pos += len(split)
            else:
                current_chunk += split
                current_pos += len(split)

        # Add final chunk
        if current_chunk.strip():
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                metadata=metadata or {},
                chunk_index=chunk_index,
                start_pos=current_pos - len(current_chunk),
                end_pos=current_pos,
            )
            chunks.append(chunk)

        logger.info(
            f"Chunked text into {len(chunks)} chunks (avg size: {sum(len(c.content) for c in chunks) // len(chunks)} chars)"
        )
        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using separator hierarchy.

        Args:
            text: Input text
            separators: List of separators (in priority order)

        Returns:
            List of text splits
        """
        if not separators:
            # No separators left, return text as-is
            return [text]

        # Use first separator
        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by separator
        splits = text.split(separator)

        # If split produced good results, return
        if len(splits) > 1:
            return [s.strip() for s in splits if s.strip()]

        # Otherwise, try next separator
        return self._recursive_split(text, remaining_separators)

    def _get_overlap_text(self, text: str) -> str:
        """
        Get overlap portion from text.

        Args:
            text: Source text

        Returns:
            Overlap text (last N characters)
        """
        if len(text) <= self.config.chunk_overlap:
            return text

        # Try to find a good breaking point
        overlap_end = self.config.chunk_overlap

        # Look for sentence boundary
        for sep in [". ", "! ", "? ", "\n"]:
            last_sep = text[:overlap_end].rfind(sep)
            if last_sep > overlap_end // 2:  # Found a good boundary
                return text[last_sep + len(sep) :]

        # Fallback: return last N characters
        return text[-overlap_end:]

    def _force_split(self, text: str) -> List[str]:
        """
        Force split text that's too large.

        Args:
            text: Text to split

        Returns:
            List of text splits
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.config.chunk_size
            chunk = text[start:end]

            # Try to break at word boundary
            if end < len(text):
                last_space = chunk.rfind(" ")
                if last_space > self.config.chunk_size // 2:
                    chunk = text[start : start + last_space]

            chunks.append(chunk)
            start += len(chunk)

        return chunks


def create_document_chunker(config: ChunkingConfig) -> DocumentChunker:
    """
    Create document chunker.

    Args:
        config: Chunking configuration

    Returns:
        Document chunker instance
    """
    return DocumentChunker(config)
