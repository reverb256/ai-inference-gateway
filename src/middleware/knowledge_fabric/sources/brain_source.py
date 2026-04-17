"""
Brain Wiki Knowledge Source Adapter for Knowledge Fabric

Provides access to the brain wiki — a filesystem-based structured
knowledge base of wiki pages, concept notes, and Q&A entries.

Reads markdown files from ~/brain/wiki/ (flat + subdirs),
parses YAML frontmatter, scores by keyword relevance, and
returns ranked KnowledgeChunks.

Inspired by DeerFlow's FileMemoryStorage (mtime-cached JSON)
but adapted for our markdown wiki format.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core import (
    KnowledgeChunk,
    KnowledgeResult,
    SourceCapability,
    SourcePriority,
)

logger = logging.getLogger(__name__)


def _parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse YAML-like frontmatter from markdown. Simple regex approach."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return {}, content

    raw = match.group(1)
    meta: Dict[str, Any] = {}
    for line in raw.strip().split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lower()
            val = val.strip().strip('"').strip("'")
            # Handle list-like tags
            if val.startswith("[") and val.endswith("]"):
                val = [v.strip().strip('"') for v in val[1:-1].split(",")]
            meta[key] = val

    body = content[match.end():]
    return meta, body


def _tokenize(text: str) -> Set[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return set(re.findall(r"[a-z0-9]{2,}", text.lower()))


def _score_relevance(query_tokens: Set[str], doc_tokens: Set[str]) -> float:
    """Score document relevance by token overlap (Jaccard-like)."""
    if not query_tokens or not doc_tokens:
        return 0.0
    overlap = len(query_tokens & doc_tokens)
    if overlap == 0:
        return 0.0
    # Favor precision (how much of query is covered)
    precision = overlap / len(query_tokens)
    # Small bonus for recall
    recall = overlap / len(doc_tokens) if doc_tokens else 0
    return min(1.0, precision * 0.8 + recall * 0.2)


@dataclass
class BrainWikiSource:
    """
    Brain wiki knowledge source.

    Reads markdown wiki pages from the brain storage root and returns
    relevant chunks based on keyword matching.

    The brain wiki is a filesystem-based knowledge base with:
    - Flat .md files in wiki/ (overview.md, model-assignment-matrix.md)
    - Concept pages in wiki/concepts/
    - Q&A pages in wiki/qa/
    - Domain pages in wiki/domains/

    Files are cached by mtime (inspired by DeerFlow's FileMemoryStorage).
    """

    brain_wiki_path: str = str(Path.home() / "brain" / "wiki")
    max_results: int = 5
    max_chunk_chars: int = 2000
    name: str = "brain_wiki"
    description: str = "Brain wiki knowledge base"
    priority: SourcePriority = SourcePriority.HIGH
    capabilities: SourceCapability = (
        SourceCapability.FACTUAL
        | SourceCapability.CONTEXTUAL
        | SourceCapability.PROCEDURAL
        | SourceCapability.COMPARATIVE
    )
    enabled: bool = True
    # Subdirectories to index (order = priority)
    subdirs: List[str] = field(default_factory=lambda: [
        "",           # root wiki files
        "concepts",   # concept pages
        "qa",         # Q&A pages
        "domains",    # domain overviews
    ])

    def __post_init__(self):
        self._cache: Dict[str, Tuple[Dict[str, Any], str, float]] = {}
        self._file_list_cache: Optional[List[Tuple[str, float]]] = None
        self._file_list_mtime: float = 0.0

    def _scan_wiki_files(self) -> List[Tuple[str, float]]:
        """Scan wiki directory for .md files, sorted by recency."""
        wiki_path = Path(self.brain_wiki_path)
        if not wiki_path.exists():
            logger.warning(f"Brain wiki path does not exist: {wiki_path}")
            return []

        # Check if directory mtime changed (need rescan)
        try:
            dir_mtime = wiki_path.stat().st_mtime
        except OSError:
            dir_mtime = 0.0

        if (
            self._file_list_cache is not None
            and dir_mtime <= self._file_list_mtime
        ):
            return self._file_list_cache

        files: List[Tuple[str, float]] = []
        seen: Set[str] = set()

        # Process subdirs in priority order
        for subdir in self.subdirs:
            target = wiki_path / subdir if subdir else wiki_path
            if not target.exists():
                continue

            for f in target.glob("*.md"):
                if f.name in seen:
                    continue
                seen.add(f.name)
                try:
                    mtime = f.stat().st_mtime
                    files.append((str(f), mtime))
                except OSError:
                    continue

        # Sort by mtime descending (newer = more relevant)
        files.sort(key=lambda x: x[1], reverse=True)
        self._file_list_cache = files
        self._file_list_mtime = dir_mtime

        return files

    def _load_file(self, filepath: str) -> Optional[Tuple[Dict[str, Any], str]]:
        """Load and cache a wiki file (mtime-based cache)."""
        try:
            current_mtime = Path(filepath).stat().st_mtime
        except OSError:
            return None

        cached = self._cache.get(filepath)
        if cached and cached[2] == current_mtime:
            return (cached[0], cached[1])

        try:
            content = Path(filepath).read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(content)
            self._cache[filepath] = (meta, body, current_mtime)
            return (meta, body)
        except (OSError, UnicodeDecodeError) as e:
            logger.debug(f"Failed to read {filepath}: {e}")
            return None

    async def retrieve(
        self,
        query: str,
        context: Optional[Dict] = None,
        **kwargs,
    ) -> KnowledgeResult:
        """
        Search brain wiki for relevant pages.

        Uses keyword overlap scoring (no embeddings required).
        Returns top-N pages as KnowledgeChunks.
        """
        import time
        start = time.time()

        query_tokens = _tokenize(query)
        files = self._scan_wiki_files()

        scored: List[Tuple[float, str, Dict[str, Any], str]] = []

        for filepath, mtime in files:
            loaded = self._load_file(filepath)
            if loaded is None:
                continue

            meta, body = loaded

            # Build searchable text from meta + body
            title = meta.get("title", Path(filepath).stem)
            summary = meta.get("summary", "")
            tags = meta.get("tags", [])
            tags_str = " ".join(tags) if isinstance(tags, list) else str(tags)

            searchable = f"{title} {summary} {tags_str} {body[:500]}"
            doc_tokens = _tokenize(searchable)

            score = _score_relevance(query_tokens, doc_tokens)

            # Boost recency (newer files get a small boost)
            age_hours = (time.time() - mtime) / 3600
            recency_boost = max(0.0, 1.0 - (age_hours / 168)) * 0.1  # 1-week decay
            score += recency_boost

            if score > 0.05:  # Minimum threshold
                scored.append((score, filepath, meta, body))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Take top N
        chunks = []
        for score, filepath, meta, body in scored[: self.max_results]:
            title = meta.get("title", Path(filepath).stem)
            summary = meta.get("summary", "")

            # Truncate body for context injection
            body_truncated = body[: self.max_chunk_chars]
            if len(body) > self.max_chunk_chars:
                body_truncated += "\n... (truncated)"

            content = f"# {title}"
            if summary:
                content += f"\n> {summary}"
            content += f"\n\n{body_truncated}"

            rel_path = os.path.relpath(filepath, self.brain_wiki_path)

            chunk = KnowledgeChunk(
                content=content,
                source=self.name,
                score=score,
                metadata={
                    "file_path": rel_path,
                    "title": title,
                    "summary": summary,
                    "tags": meta.get("tags", []),
                    "created": meta.get("createdat", meta.get("created", "")),
                    "updated": meta.get("updatedat", meta.get("updated", "")),
                    "source_type": "brain_wiki",
                },
                capabilities=self.capabilities,
            )
            chunks.append(chunk)

        retrieval_time = time.time() - start

        return KnowledgeResult(
            source_name=self.name,
            chunks=chunks,
            query=query,
            retrieval_time=retrieval_time,
            metadata={
                "total_wiki_files": len(files),
                "scored_files": len(scored),
                "returned": len(chunks),
                "scoring": "keyword_overlap",
                "cache_size": len(self._cache),
            },
        )


def create_brain_source(
    brain_wiki_path: Optional[str] = None,
    max_results: int = 5,
) -> BrainWikiSource:
    """Factory function to create brain wiki knowledge source."""
    return BrainWikiSource(
        brain_wiki_path=brain_wiki_path or str(
            Path.home() / "brain" / "wiki"
        ),
        max_results=max_results,
    )
