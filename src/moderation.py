"""
Content Moderation Service

Detects and filters harmful content including:
- Hate speech and harassment
- Violence and threats
- Sexual content
- Self-harm and suicide
- Jailbreak attempts and prompt injection
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ModerationCategory(Enum):
    """Content moderation categories."""

    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    SELF_HARM = "self_harm"
    JAILBREAK = "jailbreak"
    PROMPT_INJECTION = "prompt_injection"
    SPAM = "spam"


@dataclass
class ModerationResult:
    """Result of content moderation check."""

    flagged: bool
    categories: List[ModerationCategory] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def safe(self) -> bool:
        """Check if content is safe (not flagged)."""
        return not self.flagged


@dataclass
class ModerationPattern:
    """Moderation pattern definition."""

    category: ModerationCategory
    patterns: List[str]  # Regex patterns
    severity: float = 1.0  # 0-1 score multiplier
    description: str = ""


class ContentModerator:
    """
    Detect harmful content using pattern matching and heuristics.

    Production systems should use dedicated moderation APIs (OpenAI, etc.).
    This implementation provides baseline protection.
    """

    # Default moderation patterns
    DEFAULT_PATTERNS = [
        # Jailbreak attempts
        ModerationPattern(
            category=ModerationCategory.JAILBREAK,
            patterns=[
                r"ignore\s+(all\s+)?(previous\s+)?instructions",
                r"disregard\s+(all\s+)?(previous\s+)?instructions",
                r"forget\s+(everything|all\s+instructions)",
                r"you\s+are\s+now\s+(unrestricted|freed)",
                r"override\s+(your\s+)?programming",
                r"bypass\s+(safety\s+)?filters?",
                r"act\s+as\s+(if\s+)?you?\s+(are\s+)?not\s+(an?\s+)?AI",
                r"pretend\s+(you\s+are|to\s+be)",
                r"simulation\s+mode",
                r"DAN\s+(mode|protocol)",
            ],
            severity=1.0,
            description="Jailbreak and override attempts",
        ),
        # Prompt injection
        ModerationPattern(
            category=ModerationCategory.PROMPT_INJECTION,
            patterns=[
                r"new\s+(instruction|task|prompt)",
                r"instead\s+(of\s+)?(the\s+)?(above|previous)",
                r"actually,\s+(please\s+)?",
                r"wait,\s+(forget\s+that|let\s+me\s+rephrase)",
                r"on\s+second\s+thought",
                r"change\s+of\s+plans?",
                r"\[SYSTEM\]",
                r"\[ADMIN\]",
                r"\[DEVELOPER\]",
            ],
            severity=0.8,
            description="Prompt injection attempts",
        ),
        # Hate speech (simplified patterns)
        ModerationPattern(
            category=ModerationCategory.HATE_SPEECH,
            patterns=[
                r"\b(hate|kill|destroy)\s+(all\s+)?(?!yourself)",
                r"\b(nazi|white\s+supremacist|terrorist)",
                r"\b(discriminat|persecut)\w+",
            ],
            severity=1.0,
            description="Hate speech (basic detection)",
        ),
        # Violence
        ModerationPattern(
            category=ModerationCategory.VIOLENCE,
            patterns=[
                r"\b(bomb|explosive|weapon|gun|shoot|stab)\b",
                r"\b(murder|assault|attack)\b",
                r"\b(threaten|threat)\s+to\s+(kill|hurt|harm)",
            ],
            severity=0.9,
            description="Violence and threats",
        ),
        # Self-harm
        ModerationPattern(
            category=ModerationCategory.SELF_HARM,
            patterns=[
                r"\b(kill\s+myself|commit\s+suicide|end\s+my\s+life)\b",
                r"\b(hurt\s+myself|self\s+harm)\b",
                r"\b(suicidal|want\s+to\s+die)\b",
            ],
            severity=1.0,
            description="Self-harm and suicide",
        ),
        # Sexual content
        ModerationPattern(
            category=ModerationCategory.SEXUAL,
            patterns=[
                r"\b(pornography|explicit|nsfw)\b",
            ],
            severity=0.8,
            description="Sexual content",
        ),
        # Spam
        ModerationPattern(
            category=ModerationCategory.SPAM,
            patterns=[
                r"(click\s+here|buy\s+now|win\s+\$\w+)",
                r"http[s]?://\S{30,}",  # Very long URLs
                r"(.)\1{10,}",  # Repeated characters
            ],
            severity=0.5,
            description="Spam and suspicious patterns",
        ),
    ]

    def __init__(
        self,
        patterns: Optional[List[ModerationPattern]] = None,
        threshold: float = 0.7,
        strictness: str = "medium",  # low, medium, high
    ):
        """
        Initialize content moderator.

        Args:
            patterns: Custom moderation patterns
            threshold: Flag threshold (0-1)
            strictness: Strictness level (affects threshold)
        """
        self.patterns = patterns or self.DEFAULT_PATTERNS

        # Adjust threshold based on strictness
        strictness_thresholds = {"low": 0.9, "medium": 0.7, "high": 0.5}
        self.threshold = strictness_thresholds.get(strictness, threshold)

        # Compile regex patterns
        self._compiled_patterns = []
        for pattern_def in self.patterns:
            compiled = [
                (re.compile(pattern, re.IGNORECASE), pattern_def)
                for pattern in pattern_def.patterns
            ]
            self._compiled_patterns.extend(compiled)

        logger.info(
            f"ContentModerator initialized: "
            f"threshold={self.threshold}, strictness={strictness}, "
            f"patterns={len(self.patterns)}"
        )

    def moderate(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> ModerationResult:
        """
        Moderate content for harmful material.

        Args:
            text: Input text to moderate
            context: Optional context (user_id, session_id, etc.)

        Returns:
            ModerationResult with flag status
        """
        if not text:
            return ModerationResult(flagged=False)

        categories_found = []
        scores = {}
        details = {"matches": []}

        for pattern, pattern_def in self._compiled_patterns:
            matches = pattern.findall(text)

            if matches:
                category = pattern_def.category

                if category not in categories_found:
                    categories_found.append(category)

                # Calculate score based on severity and match count
                base_score = pattern_def.severity
                match_count = len(matches)
                score = min(1.0, base_score * (1 + match_count * 0.1))

                scores[category.value] = max(scores.get(category.value, 0), score)

                details["matches"].append(
                    {
                        "category": category.value,
                        "pattern": pattern.pattern[:50],  # Truncate for logging
                        "match_count": match_count,
                        "severity": pattern_def.severity,
                    }
                )

        # Calculate overall score
        overall_score = max(scores.values()) if scores else 0.0

        # Determine if flagged
        flagged = overall_score >= self.threshold

        result = ModerationResult(
            flagged=flagged, categories=categories_found, scores=scores, details=details
        )

        if flagged:
            logger.warning(
                f"Content flagged by moderation: "
                f"categories={[c.value for c in categories_found]}, "
                f"score={overall_score:.2f}"
            )

        return result

    def moderate_messages(
        self, messages: List[Dict[str, str]], context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, str]], ModerationResult]:
        """
        Moderate chat messages.

        Args:
            messages: List of message dicts
            context: Optional context

        Returns:
            Tuple of (filtered_messages, moderation_result)
        """
        # Check all user messages
        combined_text = "\n".join(
            msg.get("content", "") for msg in messages if msg.get("role") == "user"
        )

        result = self.moderate(combined_text, context)

        # If flagged, filter the messages
        if result.flagged:
            filtered = [msg for msg in messages if msg.get("role") != "user"]

            # Add system message about filtering
            filtered.insert(
                0,
                {
                    "role": "system",
                    "content": "[Some user messages were filtered due to content policy violations]",
                },
            )

            return filtered, result

        return messages, result

    def is_safe(self, text: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if content is safe (quick check).

        Args:
            text: Input text
            context: Optional context

        Returns:
            True if content passes moderation
        """
        result = self.moderate(text, context)
        return result.safe

    def get_categories(self) -> List[Dict[str, Any]]:
        """
        Get information about moderation categories.

        Returns:
            List of category info
        """
        categories = {}

        for pattern_def in self.patterns:
            if pattern_def.category.value not in categories:
                categories[pattern_def.category.value] = {
                    "name": pattern_def.category.value,
                    "description": pattern_def.description,
                    "severity": pattern_def.severity,
                    "pattern_count": len(pattern_def.patterns),
                }

        return list(categories.values())


# Singleton instance
_default_moderator: Optional[ContentModerator] = None


def get_default_moderator() -> ContentModerator:
    """Get or create default content moderator."""
    global _default_moderator
    if _default_moderator is None:
        _default_moderator = ContentModerator()
    return _default_moderator


def moderate_content(text: str, **kwargs) -> ModerationResult:
    """
    Moderate content using default moderator.

    Convenience function.

    Args:
        text: Input text
        **kwargs: Arguments passed to moderator

    Returns:
        ModerationResult
    """
    moderator = get_default_moderator()
    return moderator.moderate(text, **kwargs)
