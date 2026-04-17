"""
Semantic routing for query classification and source selection.

Analyzes user queries to determine intent and selects appropriate
knowledge sources based on capabilities and priorities.
"""

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from .core import KnowledgeSource, SourceCapability

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Classification of user query intent."""
    CODE = "code"                    # Code search, implementations
    FACTUAL = "factual"              # Specific facts, definitions
    PROCEDURAL = "procedural"        # How-to, tutorials, steps
    REALTIME = "realtime"            # Current data, news
    COMPARATIVE = "comparative"      # X vs Y, alternatives
    CONTEXTUAL = "contextual"        # Deep explanations, context
    UNKNOWN = "unknown"              # Unclear intent


@dataclass
class RoutingDecision:
    """
    Result of semantic routing analysis.

    Attributes:
        intent: The classified query intent
        confidence: How confident we are (0-1)
        required_capabilities: Capabilities needed to answer
        selected_sources: Chosen sources (ordered by priority)
        reasoning: Why this decision was made
    """
    intent: QueryIntent
    confidence: float
    required_capabilities: SourceCapability
    selected_sources: List[str]
    reasoning: str


class SemanticRouter:
    """
    Classifies queries and selects appropriate knowledge sources.

    Uses pattern matching heuristics to determine query intent,
    then selects sources based on capabilities and priorities.
    """

    # Minimum confidence threshold for routing
    MIN_CONFIDENCE = 0.5

    # Patterns for each intent type
    PATTERNS = {
        QueryIntent.CODE: [
            r'\b(?:function|class|method|def|import|include)\b',
            r'\b(?:code|implement|API|endpoint|interface)\b',
            r'\b(?:bug|error|exception|stack trace)\b',
            r'\b(?:refactor|optimize|debug)\b',
            r'\b(?:how do I write|how to code|show me code)\b',
            r'\b(?:snippet|example code)\b',
        ],
        QueryIntent.FACTUAL: [
            r'\b(?:what is|define|explain|describe)\b',
            r'\b(?:who is|what are|when did|where is)\b',
            r'\b(?:meaning of|definition of)\b',
        ],
        QueryIntent.PROCEDURAL: [
            r'\b(?:how do I|how to|step by step|guide)\b',
            r'\b(?:tutorial|walkthrough|instructions)\b',
            r'\b(?:setup|configure|install)\b',
            r'\b(?:process for|way to)\b',
        ],
        QueryIntent.REALTIME: [
            r'\b(?:current|latest|recent|today|now)\b',
            r'\b(?:news|weather|price|status)\b',
            r'\b(?:live|right now|happening)\b',
        ],
        QueryIntent.COMPARATIVE: [
            r'\b(?:vs|versus|compare|difference|better than)\b',
            r'\b(?:alternative|instead of|or)\b',
            r'\b(?:X vs Y|A or B)\b',
        ],
        QueryIntent.CONTEXTUAL: [
            r'\b(?:why does|how does|understand)\b',
            r'\b(?:concept|principle|theory)\b',
            r'\b(?:background|context|overview)\b',
        ],
    }

    # Compile patterns for efficiency
    _compiled_patterns = None

    @classmethod
    def _get_compiled_patterns(cls) -> Dict[QueryIntent, List[re.Pattern]]:
        if cls._compiled_patterns is None:
            cls._compiled_patterns = {
                intent: [re.compile(p, re.IGNORECASE) for p in patterns]
                for intent, patterns in cls.PATTERNS.items()
            }
        return cls._compiled_patterns

    def __init__(
        self,
        sources: List[KnowledgeSource],
        confidence_threshold: float = MIN_CONFIDENCE
    ):
        """
        Initialize router with available knowledge sources.

        Args:
            sources: All available knowledge sources (used for selection)
            confidence_threshold: Minimum confidence for routing (0-1)
        """
        self.sources_by_name = {s.name: s for s in sources}
        self.sources_by_priority = sorted(sources, key=lambda s: s.priority)
        self.confidence_threshold = confidence_threshold

    def classify(self, query: str, history: Optional[List] = None) -> RoutingDecision:
        """
        Classify query intent and select appropriate sources.

        Args:
            query: The user's query text
            history: Optional conversation history for context

        Returns:
            RoutingDecision with intent and selected sources
        """
        query_lower = query.lower().strip()
        patterns = self._get_compiled_patterns()

        # Count pattern matches for each intent
        intent_scores = {}
        for intent, regex_list in patterns.items():
            score = 0
            for regex in regex_list:
                if regex.search(query):
                    score += 1
            if score > 0:
                intent_scores[intent] = score

        # Determine primary intent
        if intent_scores:
            primary_intent = max(intent_scores, key=intent_scores.get)
            max_score = intent_scores[primary_intent]
            confidence = min(0.9, max_score * 0.2)  # Scale to 0-1
        else:
            primary_intent = QueryIntent.UNKNOWN
            confidence = 0.3

        # Map intent to required capabilities
        capability_map = {
            QueryIntent.CODE: SourceCapability.CODE,
            QueryIntent.FACTUAL: SourceCapability.FACTUAL,
            QueryIntent.PROCEDURAL: SourceCapability.PROCEDURAL,
            QueryIntent.REALTIME: SourceCapability.REALTIME,
            QueryIntent.COMPARATIVE: SourceCapability.COMPARATIVE,
            QueryIntent.CONTEXTUAL: SourceCapability.CONTEXTUAL,
        }

        required_caps = capability_map.get(primary_intent, SourceCapability.FACTUAL)

        # Select sources that can handle this intent
        selected = self._select_sources(required_caps, query)

        # Build reasoning
        reasoning = f"Intent: {primary_intent.value} (confidence: {confidence:.2f}). "
        reasoning += f"Required capabilities: {[c.name for c in SourceCapability if c in required_caps]}. "
        reasoning += f"Selected {len(selected)} sources: {selected}"

        # Apply confidence threshold - skip retrieval if too uncertain
        if confidence < self.confidence_threshold:
            logger.info(
                f"Confidence {confidence:.2f} below threshold "
                f"{self.confidence_threshold:.2f} - skipping routing"
            )
            return RoutingDecision(
                intent=primary_intent,
                confidence=confidence,
                required_capabilities=required_caps,
                selected_sources=[],  # Empty = no retrieval
                reasoning=reasoning + " | SKIPPED: low confidence"
            )

        return RoutingDecision(
            intent=primary_intent,
            confidence=confidence,
            required_capabilities=required_caps,
            selected_sources=selected,
            reasoning=reasoning
        )

    def _select_sources(
        self,
        required_capabilities: SourceCapability,
        query: str
    ) -> List[str]:
        """
        Select which sources to query for a given intent.

        Selection criteria:
        1. Source must have required capabilities
        2. Prefer higher priority (lower priority number)
        3. Source must be enabled
        4. Limit to reasonable number (top 3 per priority level)

        Args:
            required_capabilities: Capabilities needed
            query: Original query (for logging)

        Returns:
            List of source names ordered by priority
        """
        selected = []

        # Group sources by priority level
        by_priority: dict[int, List[KnowledgeSource]] = {}
        for source in self.sources_by_priority:
            if not source.enabled:
                continue
            if source.can_handle(required_capabilities):
                by_priority.setdefault(source.priority, []).append(source)

        # Select from each priority level (up to limit)
        per_priority_limit = 3
        for priority in sorted(by_priority.keys()):
            sources_at_level = by_priority[priority][:per_priority_limit]
            selected.extend(s.name for s in sources_at_level)

            # Stop if we have enough sources
            if len(selected) >= 5:  # Max total sources
                break

        logger.debug(f"Selected sources: {selected} for capabilities {required_capabilities}")
        return selected


def create_router(
    sources: List[KnowledgeSource],
    confidence_threshold: float = SemanticRouter.MIN_CONFIDENCE
) -> SemanticRouter:
    """Factory function to create a SemanticRouter.

    Args:
        sources: All available knowledge sources
        confidence_threshold: Minimum confidence for routing (0-1)

    Returns:
        Configured SemanticRouter instance
    """
    return SemanticRouter(sources, confidence_threshold=confidence_threshold)
