"""
Code Search Knowledge Source Adapter for Knowledge Fabric

Provides code search integration using SearXNG with code-specific engines.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from ..core import (
    KnowledgeChunk,
    KnowledgeResult,
    SourceCapability,
    SourcePriority,
)

logger = logging.getLogger(__name__)


@dataclass
class CodeSearchKnowledgeSource:
    """
    Code search knowledge source.

    Provides code search through SearXNG with code-specific engines
    (GitHub, GitLab, StackOverflow, etc.).
    """
    searxng_url: str = "http://10.4.98.141:7777"
    max_results: int = 5
    timeout: float = 30.0
    search_paths: List[str] = field(default_factory=list)
    name: str = "code_search"
    description: str = "Code search via SearXNG"
    priority: SourcePriority = SourcePriority.CRITICAL
    capabilities: SourceCapability = (
        SourceCapability.CODE |
        SourceCapability.PROCEDURAL
    )
    enabled: bool = True

    code_engines: List[str] = field(default_factory=lambda: [
        "github", "gitlab", "stackoverflow", "codasearch"
    ])

    def can_handle(self, capabilities: 'SourceCapability') -> bool:
        """Check if this source has the required capabilities."""
        return bool(self.capabilities & capabilities)

    def _score_code_result(self, result: Dict, query: str) -> float:
        """Score code search result quality (0-1)."""
        score = 0.0
        url = result.get("url", "").lower()
        title = result.get("title", "").lower()
        content = result.get("content", result.get("snippet", "")).lower()
        query_lower = query.lower()

        trusted_domains = [
            "github.com", "gitlab.com", "stackoverflow.com",
            "docs.rs", "readthedocs.io", "pydocs.io"
        ]
        if any(domain in url for domain in trusted_domains):
            score += 0.4

        query_words = [w for w in query_lower.split() if len(w) > 3]
        if any(word in title for word in query_words):
            score += 0.3

        if any(word in content for word in query_words):
            score += 0.2

        if url.startswith("https://"):
            score += 0.1

        if len(content) > 200:
            score += 0.1

        return min(score, 1.0)

    async def retrieve(self, query: str, **kwargs) -> KnowledgeResult:
        """
        Execute code search via SearXNG with code-specific engines.

        Returns relevant code snippets from code repositories and documentation.
        """
        import time
        start = time.time()

        sanitized_query = query[:500]

        chunks = []
        metadata = {
            "tool": "code_search",
            "type": "searxng_code_search",
            "engines": self.code_engines,
        }

        try:
            params = {
                "q": sanitized_query,
                "format": "json",
                "engines": ",".join(self.code_engines),
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    self.searxng_url,
                    params=params,
                    headers={
                        "X-Forwarded-For": "10.0.0.102",
                        "X-Real-IP": "10.0.0.102",
                    },
                )

                if response.status_code == 200:
                    data = response.json()

                    if data.get("results"):
                        results = data["results"]

                        # Score results
                        for result in results:
                            result["_score"] = self._score_code_result(result, query)

                        # Sort by score
                        results.sort(key=lambda r: r.get("_score", 0), reverse=True)

                        # Create chunks from top results
                        for idx, result in enumerate(results[:self.max_results]):
                            title = result.get("title", "")
                            snippet = result.get("content", result.get("snippet", ""))
                            url = result.get("url", "")
                            score = result.get("_score", 1.0 - (idx * 0.1))

                            content_text = f"# {title}\n\n{snippet}"

                            chunk = KnowledgeChunk(
                                content=content_text,
                                source=self.name,
                                score=score,
                                metadata={
                                    "url": url,
                                    "title": title,
                                    "engine": result.get("engine", ""),
                                    "quality_score": result.get("_score", 0.5),
                                    "language": self._detect_language(url, title),
                                },
                                capabilities=self.capabilities,
                            )
                            chunks.append(chunk)

                        metadata["total_results"] = len(chunks)
                        metadata["scored"] = True
                    else:
                        metadata["total_results"] = 0
                        metadata["note"] = "No results from SearXNG"
                else:
                    metadata["error"] = f"HTTP {response.status_code}"
                    metadata["error_type"] = "http_error"
                    logger.warning(f"Code search returned status {response.status_code}")

        except httpx.ConnectError as e:
            metadata["error"] = "Cannot connect to SearXNG"
            metadata["error_type"] = "connection_error"
            logger.error(f"Code search connection error: {e}")

        except httpx.TimeoutException:
            metadata["error"] = "Request timeout"
            metadata["error_type"] = "timeout"
            logger.error(f"Code search timeout after {self.timeout}s")

        except Exception as e:
            metadata["error"] = str(e)
            metadata["error_type"] = type(e).__name__
            logger.exception(f"Code search unexpected error: {e}")

        retrieval_time = time.time() - start

        return KnowledgeResult(
            source_name=self.name,
            chunks=chunks,
            query=query,
            retrieval_time=retrieval_time,
            metadata=metadata,
        )

    def _detect_language(self, url: str, title: str) -> str:
        """Detect programming language from URL or title."""
        url_lower = url.lower()
        title_lower = title.lower()

        # GitHub/GitLab file extensions
        lang_extensions = {
            ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
            ".go": "Go", ".rs": "Rust", ".rb": "Ruby",
            ".java": "Java", ".c": "C", ".cpp": "C++",
            ".nix": "Nix", ".sh": "Shell", ".yaml": "YAML",
            ".json": "JSON", ".md": "Markdown",
        }

        for ext, lang in lang_extensions.items():
            if ext in url_lower or ext in title_lower:
                return lang

        return "Unknown"


def create_code_search_source(
    searxng_url: str = "http://10.4.98.141:7777",
    max_results: int = 5,
) -> CodeSearchKnowledgeSource:
    """Factory function to create code search knowledge source."""
    return CodeSearchKnowledgeSource(
        searxng_url=searxng_url,
        max_results=max_results,
    )

logger = logging.getLogger(__name__)


@dataclass
class CodeSearchKnowledgeSource:
    """
    Code search knowledge source using local grep/ripgrep.
    
    Searches code files in configured paths using literal + regex search.
    """
    search_url: str = "http://127.0.0.1:8080/mcp/call"
    max_results: int = 5
    timeout: float = 30.0
    search_paths: List[str] = field(default_factory=lambda: ["/etc/nixos", "/data/projects/own"])
    name: str = "code_search"
    description: str = "Local code search via grep"
    priority: SourcePriority = SourcePriority.CRITICAL
    capabilities: SourceCapability = (
        SourceCapability.CODE |
        SourceCapability.PROCEDURAL
    )
    enabled: bool = True

    def can_handle(self, capabilities: 'SourceCapability') -> bool:
        """Check if this source has the required capabilities."""
        return bool(self.capabilities & capabilities)

    async def _run_grep(self, query: str, search_type: str = "literal") -> List[str]:
        """Run grep command and return results."""
        # Sanitize query for safety
        safe_query = query.replace("'", "\\'").replace(";", "\\;")[:200]
        
        # Determine grep options based on search type
        if search_type == "regex":
            cmd = f"grep -rn -E '{safe_query}' {' '.join(self.search_paths)} 2>/dev/null | head -{self.max_results}"
        else:
            cmd = f"grep -rn -F '{safe_query}' {' '.join(self.search_paths)} 2>/dev/null | head -{self.max_results}"
        
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
            
            results = stdout.decode().strip().split('\n') if stdout else []
            return [r for r in results if r.strip()]
        except asyncio.TimeoutError:
            logger.warning(f"Code search timed out for query: {query}")
            return []
        except Exception as e:
            logger.warning(f"Code search error: {e}")
            return []

    async def retrieve(self, query: str, **kwargs) -> KnowledgeResult:
        """
        Execute code search via local grep/ripgrep.

        Returns relevant code snippets from local filesystem.
        """
        import time
        start = time.time()

        sanitized_query = query[:500]
        
        # Determine search type from query
        search_type = "regex" if any(c in sanitized_query for c in ['.*', '\\d', '\\w', '[', ']']) else "literal"
        
        chunks = []
        metadata = {
            "tool": "code_search",
            "type": "local_grep",
            "search_paths": self.search_paths,
        }

        try:
            results = await self._run_grep(sanitized_query, search_type)
            
            # Parse grep results into chunks
            for i, result in enumerate(results[:self.max_results]):
                if ':' in result:
                    parts = result.split(':', 2)
                    if len(parts) >= 3:
                        file_path = parts[0]
                        line_num = parts[1]
                        content = parts[2]
                        
                        chunks.append({
                            "text": content.strip()[:500],
                            "source": f"{file_path}:{line_num}",
                            "score": 1.0 - (i * 0.1),  # Decreasing score for later results
                            "type": "code",
                        })
            
            metadata["total_results"] = len(chunks)
            metadata["note"] = f"Found {len(chunks)} results via local grep"
            logger.info(f"Code search: query='{sanitized_query}', results={len(chunks)}")

        except Exception as e:
            metadata["error"] = str(e)
            metadata["error_type"] = type(e).__name__
            logger.exception(f"Code search error: {e}")

        retrieval_time = time.time() - start

        return KnowledgeResult(
            source_name=self.name,
            chunks=chunks,
            query=query,
            retrieval_time=retrieval_time,
            metadata=metadata,
        )


def create_code_search_source(
    search_url: str = "http://127.0.0.1:8080/mcp/call",
    max_results: int = 5,
    search_paths: List[str] = None,
) -> CodeSearchKnowledgeSource:
    """Factory function to create code search knowledge source."""
    return CodeSearchKnowledgeSource(
        search_url=search_url,
        max_results=max_results,
        search_paths=search_paths or ["/etc/nixos", "/data/projects/own"],
    )