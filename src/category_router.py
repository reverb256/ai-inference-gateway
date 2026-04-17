"""
Category-Based Router for AI Inference Gateway.

Inspired by oh-my-opencode, this module provides intelligent model selection
based on task categories, complexity levels, and optional client hints.

Categories:
- quick: Fast, lightweight tasks (config edits, simple queries)
- ultrabrain: Deep logical reasoning, complex architecture decisions
- deep: Complex algorithms, business logic, architecture
- unspecified-high: High uncertainty, needs high quality models
- unspecified-low: Medium complexity with clear requirements
- visual-engineering: UI/UX, design systems (requires vision models)
- artistry: Creative work, copywriting, documentation
- writing: Documentation, prose, technical writing

Usage:
    # Via HTTP headers
    X-Task-Category: ultrabrain
    X-Task-Complexity: high

    # Via query parameters
    ?category=deep&complexity=medium
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from ai_inference_gateway.router import (
    ModelInfo,
    TaskSpecialization,
    RouteDecision,
)

logger = logging.getLogger(__name__)


class TaskCategory(str, Enum):
    """Task categories inspired by oh-my-opencode."""

    QUICK = "quick"
    ULTRABRAIN = "ultrabrain"
    DEEP = "deep"
    UNSPECIFIED_HIGH = "unspecified-high"
    UNSPECIFIED_LOW = "unspecified-low"
    VISUAL_ENGINEERING = "visual-engineering"
    ARTISTRY = "artistry"
    WRITING = "writing"


class ComplexityLevel(str, Enum):
    """Complexity levels for model selection."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class CategoryConfig:
    """Configuration for a task category."""

    description: str
    complexity_preference: ComplexityLevel = ComplexityLevel.MEDIUM
    preferred_specializations: List[TaskSpecialization] = field(default_factory=list)
    fallback_specializations: List[TaskSpecialization] = field(default_factory=list)
    requires_vision: bool = False
    requires_long_context: bool = False
    max_latency_ms: Optional[float] = None
    min_quality_tier: int = 1


# Category configurations inspired by oh-my-opencode
CATEGORY_CONFIGS: Dict[TaskCategory, CategoryConfig] = {
    TaskCategory.QUICK: CategoryConfig(
        description="Light, fast tasks using small/fast models",
        complexity_preference=ComplexityLevel.LOW,
        preferred_specializations=[TaskSpecialization.FAST],
        fallback_specializations=[TaskSpecialization.GENERAL],
        max_latency_ms=1000,
        min_quality_tier=1,
    ),
    TaskCategory.ULTRABRAIN: CategoryConfig(
        description="Deep logical reasoning, complex architecture decisions requiring extensive analysis",
        complexity_preference=ComplexityLevel.HIGH,
        preferred_specializations=[TaskSpecialization.AGENTIC, TaskSpecialization.CODING],
        fallback_specializations=[TaskSpecialization.LARGE_CONTEXT],
        min_quality_tier=3,
    ),
    TaskCategory.DEEP: CategoryConfig(
        description="Complex business logic, algorithms, architecture",
        complexity_preference=ComplexityLevel.HIGH,
        preferred_specializations=[TaskSpecialization.CODING, TaskSpecialization.LARGE_CONTEXT],
        fallback_specializations=[TaskSpecialization.AGENTIC],
        min_quality_tier=3,
    ),
    TaskCategory.UNSPECIFIED_HIGH: CategoryConfig(
        description="Tasks that don't fit other categories, high effort required",
        complexity_preference=ComplexityLevel.HIGH,
        preferred_specializations=[TaskSpecialization.CODING, TaskSpecialization.AGENTIC],
        fallback_specializations=[TaskSpecialization.GENERAL],
        min_quality_tier=2,
    ),
    TaskCategory.UNSPECIFIED_LOW: CategoryConfig(
        description="Tasks that don't fit other categories, low effort required",
        complexity_preference=ComplexityLevel.MEDIUM,
        preferred_specializations=[TaskSpecialization.GENERAL],
        fallback_specializations=[TaskSpecialization.FAST],
        min_quality_tier=1,
    ),
    TaskCategory.VISUAL_ENGINEERING: CategoryConfig(
        description="Frontend, UI/UX, design, styling, animation - uses vision models",
        complexity_preference=ComplexityLevel.MEDIUM,
        preferred_specializations=[TaskSpecialization.VISION],
        fallback_specializations=[TaskSpecialization.GENERAL],
        requires_vision=True,
        min_quality_tier=2,
    ),
    TaskCategory.ARTISTRY: CategoryConfig(
        description="Highly creative/artistic tasks, novel ideas",
        complexity_preference=ComplexityLevel.MEDIUM,
        preferred_specializations=[TaskSpecialization.GENERAL],
        fallback_specializations=[TaskSpecialization.LARGE_CONTEXT],
        min_quality_tier=2,
    ),
    TaskCategory.WRITING: CategoryConfig(
        description="Documentation, prose, technical writing",
        complexity_preference=ComplexityLevel.MEDIUM,
        preferred_specializations=[TaskSpecialization.LARGE_CONTEXT],
        fallback_specializations=[TaskSpecialization.GENERAL],
        requires_long_context=True,
        min_quality_tier=2,
    ),
}


# Patterns for auto-detecting categories from request content
CATEGORY_PATTERNS: Dict[TaskCategory, List[str]] = {
    TaskCategory.QUICK: [
        r"\b(config|configuration|setting|env|environment variable)\b",
        r"\b(simple|quick|fast|small|minor|trivial)\b",
        r"\b(format|lint|style|refactor)\s*(code|file)\b",
        r"\b(add\s+(import|dependency|package))\b",
    ],
    TaskCategory.ULTRABRAIN: [
        r"\b(architecture|architectural|design pattern|system design)\b",
        r"\b(strategic|strategy|roadmap|planning)\b",
        r"\b(complex reasoning|deep analysis|comprehensive)\b",
        r"\b(trade-?off|evaluate options|compare approaches)\b",
        r"\b(optimize|optimization strategy|performance plan)\b",
    ],
    TaskCategory.DEEP: [
        r"\b(algorithm|data structure|complex logic)\b",
        r"\b(implement|refactor)\s+(complex|intricate|advanced)\b",
        r"\b(business logic|domain model|entity)\b",
        r"\b(database|schema|migration|query optimization)\b",
        r"\b(authentication|authorization|security|encryption)\b",
    ],
    TaskCategory.VISUAL_ENGINEERING: [
        r"\b(UI|UX|user interface|user experience)\b",
        r"\b(component|widget|element|layout)\b",
        r"\b(style|css|theme|design|animation)\b",
        r"\b(frontend|client.?side|view|template)\b",
        r"\b(responsive|mobile|desktop|tablet)\b",
    ],
    TaskCategory.ARTISTRY: [
        r"\b(creative|innovative|novel|original)\b",
        r"\b(copywriting|marketing|narrative|story)\b",
        r"\b(engaging|compelling|persuasive)\b",
    ],
    TaskCategory.WRITING: [
        r"\b(document|documentation|readme|guide|tutorial)\b",
        r"\b(explain|explanation|comment|docstring)\b",
        r"\b(prose|text|content|article)\b",
        r"\b(technical writing|documentation style)\b",
    ],
}


@dataclass
class CategoryRouteRequest:
    """Request for category-based routing."""

    category: Optional[TaskCategory] = None
    complexity: Optional[ComplexityLevel] = None
    content: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)

    # Model preferences
    preferred_models: List[str] = field(default_factory=list)
    excluded_models: List[str] = field(default_factory=list)

    # Constraints
    max_latency_ms: Optional[float] = None
    min_quality_tier: int = 1


class CategoryRouter:
    """
    Category-based router inspired by oh-my-opencode.

    Routes requests to appropriate models based on:
    1. Explicit category hint (X-Task-Category header)
    2. Complexity level (X-Task-Complexity header)
    3. Content analysis (auto-detection)
    4. Model availability and performance
    """

    def __init__(
        self,
        models: Dict[str, ModelInfo],
        default_category: TaskCategory = TaskCategory.UNSPECIFIED_LOW,
        enable_auto_detection: bool = True,
    ):
        """
        Initialize category router.

        Args:
            models: Dictionary of available models
            default_category: Default category when none specified
            enable_auto_detection: Enable content-based category detection
        """
        self.models = models
        self.default_category = default_category
        self.enable_auto_detection = enable_auto_detection

        # Build model pools by specialization
        self._build_model_pools()

        # GLM model tier configuration (for ZAI backend)
        self.glm_tiers = {
            5: ["glm-5", "glm-5-flash"],  # Highest tier
            4.7: ["glm-4.7"],
            4.6: ["glm-4.6", "glm-4.6v"],  # Vision model
            4.5: ["glm-4.5-air"],  # Fast model
        }

    def _build_model_pools(self) -> None:
        """Build model pools organized by specialization and quality tier."""
        self.model_pools: Dict[TaskSpecialization, List[ModelInfo]] = {
            spec: [] for spec in TaskSpecialization
        }

        # Also organize by quality tier (priority = quality tier)
        self.models_by_tier: Dict[int, List[ModelInfo]] = {}

        for model in self.models.values():
            # Add to specialization pools
            for spec in model.specializations:
                self.model_pools[spec].append(model)

            # Add to tier pool
            tier = model.priority
            if tier not in self.models_by_tier:
                self.models_by_tier[tier] = []
            self.models_by_tier[tier].append(model)

        # Sort pools by priority (highest first)
        for pool in self.model_pools.values():
            pool.sort(key=lambda m: m.priority, reverse=True)

        for tier in self.models_by_tier:
            self.models_by_tier[tier].sort(key=lambda m: m.priority, reverse=True)

        logger.info(
            f"Built model pools: {len(self.models)} models, "
            f"{len(self.model_pools)} specializations, "
            f"{len(self.models_by_tier)} quality tiers"
        )

    def parse_category_request(
        self,
        headers: Dict[str, str],
        query_params: Dict[str, str],
        content: Optional[str] = None,
    ) -> CategoryRouteRequest:
        """
        Parse category routing request from HTTP request.

        Args:
            headers: HTTP headers from request
            query_params: Query parameters from request
            content: Request body content for auto-detection

        Returns:
            CategoryRouteRequest with parsed values
        """
        request = CategoryRouteRequest(
            headers=headers,
            query_params=query_params,
            content=content,
        )

        # Parse category from header or query param
        category_header = headers.get("X-Task-Category") or headers.get("X-Task-Category")
        category_param = query_params.get("category")

        category_str = category_header or category_param
        if category_str:
            try:
                request.category = TaskCategory(category_str.lower())
            except ValueError:
                logger.warning(f"Invalid category: {category_str}")

        # Parse complexity from header or query param
        complexity_header = headers.get("X-Task-Complexity") or headers.get("X-Task-Complexity")
        complexity_param = query_params.get("complexity")

        complexity_str = complexity_header or complexity_param
        if complexity_str:
            try:
                request.complexity = ComplexityLevel(complexity_str.lower())
            except ValueError:
                logger.warning(f"Invalid complexity: {complexity_str}")

        # Parse model preferences
        preferred = headers.get("X-Preferred-Models") or query_params.get("preferred")
        if preferred:
            request.preferred_models = [m.strip() for m in preferred.split(",")]

        excluded = headers.get("X-Excluded-Models") or query_params.get("excluded")
        if excluded:
            request.excluded_models = [m.strip() for m in excluded.split(",")]

        # Parse constraints
        max_latency = headers.get("X-Max-Latency") or query_params.get("max_latency")
        if max_latency:
            try:
                request.max_latency_ms = float(max_latency)
            except ValueError:
                pass

        return request

    def detect_category(self, content: str) -> Optional[TaskCategory]:
        """
        Auto-detect category from request content.

        Args:
            content: Request content to analyze

        Returns:
            Detected category or None
        """
        if not content or not self.enable_auto_detection:
            return None

        content_lower = content.lower()
        scores: Dict[TaskCategory, int] = {}

        for category, patterns in CATEGORY_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content_lower))
                score += matches
            if score > 0:
                scores[category] = score

        if not scores:
            return None

        # Return category with highest score
        return max(scores.items(), key=lambda x: x[1])[0]

    def select_models_for_category(
        self,
        category: TaskCategory,
        complexity: Optional[ComplexityLevel] = None,
        request: Optional[CategoryRouteRequest] = None,
    ) -> List[ModelInfo]:
        """
        Select models for a given category and complexity.

        Args:
            category: Task category
            complexity: Optional complexity level
            request: Optional routing request with constraints

        Returns:
            List of candidate models sorted by priority
        """
        config = CATEGORY_CONFIGS.get(category)
        if not config:
            logger.warning(f"Unknown category: {category}")
            return []

        # Use complexity from request or category default
        effective_complexity = complexity or config.complexity_preference

        # Get candidate models from preferred specializations
        candidates: List[ModelInfo] = []

        for spec in config.preferred_specializations:
            if spec in self.model_pools:
                candidates.extend(self.model_pools[spec])

        # If no candidates from preferred, try fallback
        if not candidates:
            for spec in config.fallback_specializations:
                if spec in self.model_pools:
                    candidates.extend(self.model_pools[spec])

        # Filter by quality tier
        min_tier = config.min_quality_tier
        if request and request.min_quality_tier > min_tier:
            min_tier = request.min_quality_tier

        candidates = [m for m in candidates if m.priority >= min_tier]

        # Apply exclusions
        if request and request.excluded_models:
            excluded_set = set(request.excluded_models)
            candidates = [m for m in candidates if m.id not in excluded_set]

        # Apply latency constraint
        max_latency = config.max_latency_ms
        if request and request.max_latency_ms:
            max_latency = min(max_latency, request.max_latency_ms)

        if max_latency:
            candidates = [
                m for m in candidates
                if not m.estimated_tokens_per_second or
                (1000 / m.estimated_tokens_per_second) * 100 <= max_latency
            ]

        # Sort by priority and apply complexity boost
        candidates.sort(key=lambda m: m.priority, reverse=True)

        # Boost priority based on complexity
        complexity_boost = {
            ComplexityLevel.LOW: -1,
            ComplexityLevel.MEDIUM: 0,
            ComplexityLevel.HIGH: 1,
        }

        # Re-sort with complexity consideration
        # For high complexity, prefer higher tier models
        if effective_complexity == ComplexityLevel.HIGH:
            candidates.sort(key=lambda m: (-m.priority, m.id))

        return candidates

    def route(
        self,
        headers: Dict[str, str],
        query_params: Dict[str, str],
        content: Optional[str] = None,
    ) -> RouteDecision:
        """
        Route a request to the best model based on category.

        Args:
            headers: HTTP headers from request
            query_params: Query parameters from request
            content: Request body content

        Returns:
            RouteDecision with selected model
        """
        # Parse request
        request = self.parse_category_request(headers, query_params, content)

        # Determine category
        category = request.category

        if not category and content and self.enable_auto_detection:
            detected = self.detect_category(content)
            if detected:
                category = detected
                logger.debug(f"Auto-detected category: {detected.value}")

        if not category:
            category = self.default_category

        # Select models for category
        candidates = self.select_models_for_category(
            category,
            request.complexity,
            request,
        )

        # Apply preferred models if specified
        if request.preferred_models:
            preferred_set = set(request.preferred_models)
            for candidate in candidates:
                if candidate.id in preferred_set:
                    return RouteDecision(
                        model=candidate.id,
                        confidence=1.0,
                        reason=f"Preferred model for category '{category.value}'",
                        estimated_tokens=0,
                        backend=candidate.backend,
                    )

        # Select best candidate
        if candidates:
            best = candidates[0]
            return RouteDecision(
                model=best.id,
                confidence=0.9,
                reason=f"Selected for category '{category.value}' (tier {best.priority})",
                estimated_tokens=0,
                backend=best.backend,
            )

        # Fallback to any available model
        if self.models:
            fallback = next(iter(self.models.values()))
            return RouteDecision(
                model=fallback.id,
                confidence=0.5,
                reason=f"Fallback for category '{category.value}' (no matching models)",
                estimated_tokens=0,
                backend=fallback.backend,
            )

        # No models available
        raise ValueError("No models available for routing")

    def get_category_info(self) -> Dict[str, dict]:
        """
        Get information about all categories.

        Returns:
            Dictionary mapping category names to their configurations
        """
        return {
            category.value: {
                "description": config.description,
                "complexity_preference": config.complexity_preference.value,
                "preferred_specializations": [s.value for s in config.preferred_specializations],
                "fallback_specializations": [s.value for s in config.fallback_specializations],
                "requires_vision": config.requires_vision,
                "requires_long_context": config.requires_long_context,
                "min_quality_tier": config.min_quality_tier,
            }
            for category, config in CATEGORY_CONFIGS.items()
        }


def create_category_router(
    models: List[ModelInfo],
    default_category: TaskCategory = TaskCategory.UNSPECIFIED_LOW,
    enable_auto_detection: bool = True,
) -> CategoryRouter:
    """
    Create a category router with the given models.

    Args:
        models: List of available models
        default_category: Default category for unspecified requests
        enable_auto_detection: Enable content-based category detection

    Returns:
        Configured CategoryRouter instance
    """
    models_dict = {model.id: model for model in models}
    return CategoryRouter(
        models=models_dict,
        default_category=default_category,
        enable_auto_detection=enable_auto_detection,
    )
