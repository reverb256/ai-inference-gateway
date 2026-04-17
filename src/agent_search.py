"""
Agent-Optimized Search Integration

Enhanced SearXNG integration for AI agent workflows with:
- Intent detection (research, code, facts, troubleshooting)
- Context-aware query refinement
- Result summarization for LLM consumption
- Source confidence scoring
- Progressive refinement based on feedback
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from ai_inference_gateway.searxng_integration import SearxngIntegration

logger = logging.getLogger(__name__)


class SearchIntent:
    """Search intent categories for intelligent routing."""

    RESEARCH = "research"  # Deep dive, academic, comprehensive
    CODE = "code"  # Programming, implementation, syntax
    FACTS = "facts"  # Quick facts, definitions, summaries
    TROUBLESHOOTING = "troubleshooting"  # Error solving, debugging
    DISCOVERY = "discovery"  # Exploration, comparison, shopping


class AgentSearchEngine:
    """
    Enhanced search engine optimized for AI agent workflows.

    Features:
    1. Intent Detection: Understands what the agent is trying to accomplish
    2. Context-Aware Refinement: Improves queries based on conversation context
    3. Result Summarization: Condenses results for efficient LLM consumption
    4. Source Scoring: Ranks results by authority and relevance
    5. Progressive Refinement: Learns from feedback to improve results
    """

    # Intent detection patterns
    INTENT_PATTERNS = {
        SearchIntent.RESEARCH: {
            "prefixes": ["research", "academic", "scholarly", "comprehensive", "in-depth"],
            "keywords": ["paper", "study", "thesis", "analysis", "survey", "review", "comparison"],
            "questions": ["what are the latest", "current state of", "recent developments"],
        },
        SearchIntent.CODE: {
            "prefixes": ["implement", "code", "program", "develop", "create"],
            "keywords": ["function", "class", "api", "library", "framework", "module", "syntax", "example"],
            "questions": ["how to", "how do i", "implement", "code example"],
        },
        SearchIntent.FACTS: {
            "prefixes": ["define", "explain", "what is", "describe"],
            "keywords": ["definition", "meaning", "summary", "overview", "introduction"],
            "questions": ["what is", "define", "explain", "describe"],
        },
        SearchIntent.TROUBLESHOOTING: {
            "prefixes": ["fix", "error", "problem", "issue", "debug", "resolve"],
            "keywords": ["error", "bug", "fail", "crash", "exception", "not working", "broken"],
            "questions": ["why is", "how to fix", "getting error", "doesn't work"],
        },
        SearchIntent.DISCOVERY: {
            "prefixes": ["find", "search", "look for", "discover"],
            "keywords": ["best", "top", "recommend", "compare", "vs", "versus", "alternative"],
            "questions": ["which", "what's the best", "recommend", "compare"],
        },
    }

    # Domain-specific trusted sources
    TRUSTED_SOURCES = {
        SearchIntent.RESEARCH: [
            "arxiv.org", "scholar.google.com", "semanticscholar.org",
            "dl.acm.org", "ieeexplore.ieee.org", "springer.com",
            "nature.com", "science.org", "pnas.org"
        ],
        SearchIntent.CODE: [
            "github.com", "gitlab.com", "stackoverflow.com", "docs.rs",
            "developer.mozilla.org", "numpy.org", "postgresql.org",
            "pypi.org", "crates.io", "npmjs.com"
        ],
        SearchIntent.TROUBLESHOOTING: [
            "stackoverflow.com", "github.com", "gitlab.com", "reddit.com",
            "docs.rs", "developer.mozilla.org", "serverfault.com"
        ],
        SearchIntent.FACTS: [
            "wikipedia.org", "britannica.com", "nih.gov", "who.int",
            "nature.com", "science.org"
        ],
    }

    def __init__(self, searxng: SearxngIntegration):
        self.searxng = searxng
        self.query_history: List[Dict] = []  # Track past queries for context
        self.feedback_history: List[Dict] = []  # Track user feedback for learning

    def detect_intent(self, query: str, context: Optional[str] = None) -> SearchIntent:
        """
        Detect the intent of a search query.

        Args:
            query: The search query
            context: Optional conversation context for better detection

        Returns:
            SearchIntent enum value
        """
        query_lower = query.lower().strip()

        # Score each intent
        intent_scores = {}

        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0

            # Check prefixes
            for prefix in patterns["prefixes"]:
                if query_lower.startswith(prefix):
                    score += 3

            # Check keywords
            for keyword in patterns["keywords"]:
                if keyword in query_lower:
                    score += 2

            # Check question patterns
            for question in patterns["questions"]:
                if question in query_lower:
                    score += 2

            intent_scores[intent] = score

        # Consider context if provided
        if context:
            context_lower = context.lower()
            if "error" in context_lower or "debug" in context_lower:
                intent_scores[SearchIntent.TROUBLESHOOTING] += 2
            elif "code" in context_lower or "implement" in context_lower:
                intent_scores[SearchIntent.CODE] += 2
            elif "research" in context_lower or "study" in context_lower:
                intent_scores[SearchIntent.RESEARCH] += 2

        # Return intent with highest score, or FACTS as default
        max_score = max(intent_scores.values())
        if max_score > 0:
            return max(intent_scores, key=intent_scores.get)
        return SearchIntent.FACTS

    def refine_query(
        self,
        query: str,
        intent: SearchIntent,
        context: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Refine search query based on intent and context.

        Returns:
            Tuple of (refined_query, metadata about changes)
        """
        refined = query
        metadata = {"original": query, "changes": []}

        # Intent-specific refinements
        if intent == SearchIntent.RESEARCH:
            # Add academic/source terms
            if not any(term in query.lower() for term in ["paper", "study", "research"]):
                refined = f"{query} research study"
                metadata["changes"].append("added academic focus")

        elif intent == SearchIntent.CODE:
            # Add implementation focus
            if "example" not in query.lower():
                refined = f"{query} example"
                metadata["changes"].append("added example request")

            # Detect programming language from context
            if context:
                for lang in ["python", "javascript", "rust", "go", "java", "nix"]:
                    if lang in context.lower() and lang not in query.lower():
                        refined = f"{refined} {lang}"
                        metadata["changes"].append(f"added language: {lang}")
                        break

        elif intent == SearchIntent.TROUBLESHOOTING:
            # Add error/solution context
            if "error" not in query.lower() and "fix" not in query.lower():
                refined = f"{query} error fix"
                metadata["changes"].append("added error context")

            # Add "solution" or "how to"
            if not query.lower().startswith(("how", "what", "why")):
                refined = f"how to fix {query}"
                metadata["changes"].append("converted to how-to question")

        # Add site-specific constraints if appropriate
        if intent == SearchIntent.CODE and "github.com" not in refined:
            # For code searches, prefer GitHub and StackOverflow
            metadata["suggested_sites"] = ["github.com", "stackoverflow.com"]

        elif intent == SearchIntent.RESEARCH and "arxiv.org" not in refined:
            # For research, prefer academic sources
            metadata["suggested_sites"] = ["arxiv.org", "scholar.google.com"]

        return refined, metadata

    def score_result_quality(
        self,
        result: Dict[str, Any],
        intent: SearchIntent
    ) -> float:
        """
        Score a search result based on quality and relevance.

        Returns:
            Quality score from 0.0 to 1.0
        """
        score = 0.0

        # 1. Source authority (40% of score)
        url = result.get("url", "").lower()
        trusted_sources = self.TRUSTED_SOURCES.get(intent, [])

        for trusted in trusted_sources:
            if trusted in url:
                score += 0.4
                break
        else:
            # Still give some points for .edu, .org, .gov
            if any(tld in url for tld in [".edu", ".org", ".gov"]):
                score += 0.2

        # 2. Content richness (30% of score)
        content = result.get("content", result.get("snippet", ""))
        if len(content) > 200:
            score += 0.15
        if len(content) > 500:
            score += 0.15

        # 3. Query relevance (20% of score)
        title = result.get("title", "").lower()
        snippet = content.lower()

        # Exact phrase match in title
        if result.get("query", "").lower() in title:
            score += 0.15
        elif any(word in title for word in result.get("query", "").lower().split() if len(word) > 3):
            score += 0.10

        # 4. Recency for technical content (10% of score)
        if intent in [SearchIntent.CODE, SearchIntent.TROUBLESHOOTING]:
            current_year = datetime.now().year
            for year in [str(current_year), str(current_year - 1), str(current_year - 2)]:
                if year in content or year in title:
                    score += 0.10
                    break

        return min(score, 1.0)

    def summarize_for_llm(
        self,
        results: List[Dict[str, Any]],
        intent: SearchIntent,
        max_length: int = 2000
    ) -> str:
        """
        Summarize search results for efficient LLM consumption.

        Returns:
            Condensed summary with key information and sources
        """
        if not results:
            return "No relevant results found."

        # Group results by quality
        high_quality = [r for r in results if r.get("quality_score", 0) > 0.7]
        medium_quality = [r for r in results if 0.4 < r.get("quality_score", 0) <= 0.7]
        low_quality = [r for r in results if r.get("quality_score", 0) <= 0.4]

        summary_parts = []

        # Top results (high quality)
        if high_quality:
            summary_parts.append("## Top Results\n")
            for i, result in enumerate(high_quality[:3], 1):
                title = result.get("title", "Untitled")
                url = result.get("url", "")
                content = result.get("content", result.get("snippet", ""))[:300]

                summary_parts.append(
                    f"{i}. **{title}**\n"
                    f"   - URL: {url}\n"
                    f"   - Summary: {content}...\n"
                    f"   - Quality: {result.get('quality_score', 0):.2f}\n"
                )

        # Additional results
        if medium_quality:
            summary_parts.append(f"\n## Additional Results ({len(medium_quality)} items)\n")
            for result in medium_quality[:5]:
                title = result.get("title", "Untitled")
                url = result.get("url", "")
                summary_parts.append(f"- **{title}**: {url}")

        # Sources summary
        sources = {}
        for result in results:
            url = result.get("url", "")
            domain = re.sub(r"^https?://([^/]+).*$", r"\1", url)
            sources[domain] = sources.get(domain, 0) + 1

        summary_parts.append(f"\n## Sources\n")
        for domain, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            summary_parts.append(f"- {domain}: {count} result(s)")

        full_summary = "\n".join(summary_parts)

        # Truncate if necessary
        if len(full_summary) > max_length:
            full_summary = full_summary[:max_length] + "\n\n... (truncated)"

        return full_summary

    async def search_with_agent_workflow(
        self,
        query: str,
        context: Optional[str] = None,
        intent: Optional[SearchIntent] = None,
        max_results: int = 10,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform agent-optimized search with full workflow.

        This is the main entry point for AI agents.

        Args:
            query: Search query
            context: Optional conversation context
            intent: Optional pre-detected intent (auto-detected if None)
            max_results: Maximum number of results to return
            use_cache: Whether to use cached results

        Returns:
            Dict with:
            - results: List of search results with quality scores
            - intent: Detected intent
            - query_refinement: Info about how query was refined
            - summary: Condensed summary for LLM consumption
            - metadata: Additional metadata about the search
        """
        # Detect intent if not provided
        if intent is None:
            intent = self.detect_intent(query, context)

        # Refine query based on intent and context
        refined_query, refinement_metadata = self.refine_query(query, intent, context)

        # Perform search with domain routing
        search_result = await self.searxng.search_with_domain_routing(
            query=refined_query,
            domain=intent,
            max_results=max_results,
            use_cache=use_cache,
        )

        # Score results by quality
        if search_result.get("results"):
            for result in search_result["results"]:
                result["quality_score"] = self.score_result_quality(result, intent)

            # Sort by quality score
            search_result["results"].sort(
                key=lambda x: x.get("quality_score", 0), reverse=True
            )

        # Generate summary for LLM
        summary = self.summarize_for_llm(
            search_result.get("results", []),
            intent
        )

        # Track query for context
        self.query_history.append({
            "query": query,
            "refined_query": refined_query,
            "intent": intent,
            "timestamp": datetime.now().isoformat(),
            "result_count": len(search_result.get("results", [])),
        })

        # Return comprehensive result
        return {
            "results": search_result.get("results", []),
            "intent": intent,
            "query_refinement": refinement_metadata,
            "summary": summary,
            "metadata": {
                "original_query": query,
                "refined_query": refined_query,
                "engines_used": search_result.get("engines_used", []),
                "cached": search_result.get("cached", False),
                "routing": search_result.get("routing", {}),
            },
        }

    async def feedback(
        self,
        query: str,
        selected_results: List[int],
        rating: Optional[int] = None
    ):
        """
        Record feedback about search results for progressive improvement.

        Args:
            query: The original search query
            selected_results: Indices of results the user found useful
            rating: Optional rating (1-5) of overall search quality
        """
        feedback_entry = {
            "query": query,
            "selected_results": selected_results,
            "rating": rating,
            "timestamp": datetime.now().isoformat(),
        }

        self.feedback_history.append(feedback_entry)

        # Learn from feedback (simplified version)
        # In a full implementation, this would:
        # 1. Analyze patterns in selected results
        # 2. Adjust quality scoring weights
        # 3. Update query refinement strategies
        # 4. Train a lightweight model on feedback

        logger.info(f"Recorded feedback for query: {query}, rating: {rating}")

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent search learning."""
        return {
            "total_searches": len(self.query_history),
            "total_feedback": len(self.feedback_history),
            "intent_distribution": self._get_intent_distribution(),
            "recent_queries": self.query_history[-10:] if self.query_history else [],
        }

    def _get_intent_distribution(self) -> Dict[str, int]:
        """Get distribution of intents in query history."""
        distribution = {}
        for entry in self.query_history:
            intent = entry.get("intent", "unknown")
            distribution[intent] = distribution.get(intent, 0) + 1
        return distribution


# Global instance
_agent_search_engine: Optional[AgentSearchEngine] = None


def get_agent_search_engine(searxng: SearxngIntegration) -> AgentSearchEngine:
    """Get or create global agent search engine."""
    global _agent_search_engine
    if _agent_search_engine is None:
        _agent_search_engine = AgentSearchEngine(searxng)
    return _agent_search_engine
