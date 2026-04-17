"""
Self-Improvement Module for AI Inference Gateway.

Implements meta-learning from all gateway interactions:
- Episodic memory logging for requests/resolutions
- Semantic pattern extraction from routing decisions
- Self-correction triggers on errors and failures
- Feedback collection for continuous improvement
- Integration with NixOS cluster memory system
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


# Memory paths (using /run for ai-inference user access)
MEMORY_BASE = Path("/run/ai-inference/memory")
EPISODIC_DIR = MEMORY_BASE / "episodic"
WORKING_DIR = MEMORY_BASE / "working"
SEMANTIC_FILE = MEMORY_BASE / "semantic-patterns.json"


class EpisodeOutcome(str, Enum):
    """Possible outcomes for an episode"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


@dataclass
class RoutingEpisode:
    """A single routing decision episode"""
    episode_id: str
    timestamp: str
    model_requested: str
    model_routed: str
    routing_reason: str
    token_count: int
    task_type: str
    latency_ms: float
    outcome: EpisodeOutcome
    error: Optional[str] = None
    user_feedback: Optional[int] = None  # 1-10 rating
    backend_used: str = "unknown"
    retry_count: int = 0
    fallback_triggered: bool = False
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ErrorEpisode:
    """An error/failure episode for self-correction"""
    episode_id: str
    timestamp: str
    error_type: str
    error_message: str
    context: Dict[str, Any]
    resolution: Optional[str] = None
    related_pattern: Optional[str] = None
    prevented: bool = False  # If self-correction prevented a failure
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PatternCandidate:
    """A candidate pattern extracted from experience"""
    pattern_id: str
    name: str
    category: str
    confidence: float
    problem: str
    solution: str
    source_episode: str
    applications: int = 0
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return asdict(self)


class SelfImprovementEngine:
    """
    Main self-improvement engine for the gateway.
    
    Learns from:
    - Routing decisions and outcomes
    - Error patterns and resolutions
    - User feedback on quality
    - Backend performance characteristics
    """
    
    def __init__(
        self,
        memory_base: Path = MEMORY_BASE,
        enabled: bool = True,
        auto_extract_patterns: bool = True,
        min_confidence_for_update: float = 0.7,
    ):
        self.memory_base = memory_base
        self.enabled = enabled
        self.auto_extract_patterns = auto_extract_patterns
        self.min_confidence = min_confidence_for_update
        
        self.episodic_dir = memory_base / "episodic"
        self.working_dir = memory_base / "working"
        self.semantic_file = memory_base / "semantic-patterns.json"
        
        # In-memory buffers for batch writes
        self._routing_buffer: List[RoutingEpisode] = []
        self._error_buffer: List[ErrorEpisode] = []
        self._buffer_size = 10
        
        # Statistics
        self._total_episodes = 0
        self._patterns_extracted = 0
        
        # Ensure directories exist
        self.episodic_dir.mkdir(parents=True, exist_ok=True)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
    def _generate_episode_id(self) -> str:
        """Generate unique episode ID"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        hash_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"ep-{timestamp}-{hash_suffix}"
    
    def _generate_pattern_id(self) -> str:
        """Generate unique pattern ID"""
        timestamp = datetime.now().strftime("%Y%m%d")
        count = self._patterns_extracted + 1
        return f"pat-{timestamp}-{count:03d}"
    
    async def log_routing_decision(
        self,
        model_requested: str,
        model_routed: str,
        routing_reason: str,
        token_count: int,
        task_type: str,
        latency_ms: float,
        backend_used: str = "unknown",
        retry_count: int = 0,
        fallback_triggered: bool = False,
        error: Optional[str] = None,
    ) -> str:
        """Log a routing decision episode"""
        if not self.enabled:
            return ""
        
        episode = RoutingEpisode(
            episode_id=self._generate_episode_id(),
            timestamp=datetime.now().isoformat(),
            model_requested=model_requested,
            model_routed=model_routed,
            routing_reason=routing_reason,
            token_count=token_count,
            task_type=task_type,
            latency_ms=latency_ms,
            outcome=EpisodeOutcome.FAILURE if error else EpisodeOutcome.SUCCESS,
            error=error,
            backend_used=backend_used,
            retry_count=retry_count,
            fallback_triggered=fallback_triggered,
        )
        
        self._routing_buffer.append(episode)
        self._total_episodes += 1
        
        # Flush buffer if full
        if len(self._routing_buffer) >= self._buffer_size:
            await self._flush_routing_buffer()
        
        return episode.episode_id
    
    async def log_error(
        self,
        error_type: str,
        error_message: str,
        context: Dict[str, Any],
        resolution: Optional[str] = None,
    ) -> str:
        """Log an error episode for self-correction"""
        if not self.enabled:
            return ""
        
        episode = ErrorEpisode(
            episode_id=self._generate_episode_id(),
            timestamp=datetime.now().isoformat(),
            error_type=error_type,
            error_message=error_message,
            context=context,
            resolution=resolution,
        )
        
        self._error_buffer.append(episode)
        
        # Flush error immediately
        await self._flush_error_buffer()
        
        return episode.episode_id
    
    async def record_feedback(self, episode_id: str, rating: int) -> bool:
        """Record user feedback for an episode"""
        if not self.enabled:
            return False
        
        # Update in buffer if present
        for ep in self._routing_buffer:
            if ep.episode_id == episode_id:
                ep.user_feedback = rating
                return True
        
        # Otherwise update in file
        today_file = self.episodic_dir / f"{datetime.now().strftime('%Y-%m-%d')}-routing.json"
        if today_file.exists():
            try:
                data = json.loads(today_file.read_text())
                for ep in data.get("episodes", []):
                    if ep.get("episode_id") == episode_id:
                        ep["user_feedback"] = rating
                        today_file.write_text(json.dumps(data, indent=2))
                        return True
            except Exception as e:
                logger.warning(f"Failed to update feedback: {e}")
        
        return False
    
    async def _flush_routing_buffer(self) -> None:
        """Flush routing episodes to episodic memory"""
        if not self._routing_buffer:
            return
        
        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"{today}-gateway.json"
        filepath = self.episodic_dir / filename
        
        # Load existing or create new
        if filepath.exists():
            try:
                data = json.loads(filepath.read_text())
            except Exception:
                data = {"date": today, "episodes": []}
        else:
            data = {"date": today, "episodes": []}
        
        # Add new episodes
        for ep in self._routing_buffer:
            data["episodes"].append(ep.to_dict())
        
        # Write back
        try:
            filepath.write_text(json.dumps(data, indent=2))
            logger.debug(f"Logged {len(self._routing_buffer)} routing episodes to {filepath}")
        except Exception as e:
            logger.error(f"Failed to write routing episodes: {e}")
        
        self._routing_buffer.clear()
    
    async def _flush_error_buffer(self) -> None:
        """Flush error episodes to episodic memory"""
        if not self._error_buffer:
            return
        
        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"{today}-errors.json"
        filepath = self.episodic_dir / filename
        
        # Load existing or create new
        if filepath.exists():
            try:
                data = json.loads(filepath.read_text())
            except Exception:
                data = {"date": today, "episodes": []}
        else:
            data = {"date": today, "episodes": []}
        
        # Add new episodes
        for ep in self._error_buffer:
            data["episodes"].append(ep.to_dict())
        
        # Write back
        try:
            filepath.write_text(json.dumps(data, indent=2))
            logger.debug(f"Logged {len(self._error_buffer)} error episodes to {filepath}")
        except Exception as e:
            logger.error(f"Failed to write error episodes: {e}")
        
        self._error_buffer.clear()
    
    async def extract_patterns(self, min_occurrences: int = 3) -> List[PatternCandidate]:
        """
        Extract patterns from accumulated episodic memory.
        
        Analyzes:
        - Recurring error types
        - Successful routing strategies
        - Performance characteristics
        """
        if not self.enabled or not self.auto_extract_patterns:
            return []
        
        patterns = []
        
        # Analyze recent episodes for patterns
        recent_files = sorted(
            self.episodic_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:7]  # Last 7 days
        
        error_counts: Dict[str, int] = {}
        routing_stats: Dict[str, List[float]] = {}  # model -> latencies
        
        for filepath in recent_files:
            try:
                data = json.loads(filepath.read_text())
                for ep in data.get("episodes", []):
                    # Count errors
                    if ep.get("error"):
                        error_type = ep.get("error", "unknown")[:50]
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1
                    
                    # Track routing performance
                    if ep.get("outcome") == "success":
                        model = ep.get("model_routed", "unknown")
                        latency = ep.get("latency_ms", 0)
                        if model not in routing_stats:
                            routing_stats[model] = []
                        routing_stats[model].append(latency)
            except Exception as e:
                logger.warning(f"Failed to analyze {filepath}: {e}")
        
        # Extract error patterns
        for error, count in error_counts.items():
            if count >= min_occurrences:
                pattern = PatternCandidate(
                    pattern_id=self._generate_pattern_id(),
                    name=f"Handle {error[:30]}",
                    category="error_handling",
                    confidence=min(count / 10, 1.0),
                    problem=f"Recurring error: {error}",
                    solution="Add specific handling or prevention",
                    source_episode=f"analysis-{datetime.now().strftime('%Y%m%d')}",
                )
                patterns.append(pattern)
        
        # Extract performance patterns
        for model, latencies in routing_stats.items():
            if len(latencies) >= 5:  # Minimum samples
                avg_latency = sum(latencies) / len(latencies)
                if avg_latency > 5000:  # 5 seconds threshold
                    pattern = PatternCandidate(
                        pattern_id=self._generate_pattern_id(),
                        name=f"Optimize {model} routing",
                        category="performance",
                        confidence=0.7,
                        problem=f"{model} has high latency: {avg_latency:.0f}ms avg",
                        solution="Consider alternative models or pre-warming",
                        source_episode=f"analysis-{datetime.now().strftime('%Y%m%d')}",
                    )
                    patterns.append(pattern)
        
        self._patterns_extracted += len(patterns)
        return patterns
    
    async def update_semantic_memory(self, patterns: List[PatternCandidate]) -> int:
        """Update semantic memory with extracted patterns"""
        if not patterns:
            return 0
        
        # Load existing semantic memory
        if self.semantic_file.exists():
            try:
                data = json.loads(self.semantic_file.read_text())
            except Exception:
                data = {"meta": {}, "patterns": {}}
        else:
            data = {"meta": {}, "patterns": {}}
        
        # Add new patterns above confidence threshold
        added = 0
        for pattern in patterns:
            if pattern.confidence >= self.min_confidence:
                data["patterns"][pattern.pattern_id] = pattern.to_dict()
                added += 1
        
        # Update metadata
        data["meta"]["last_updated"] = datetime.now().isoformat()
        data["meta"]["total_patterns"] = len(data.get("patterns", {}))
        
        # Write back
        try:
            self.semantic_file.write_text(json.dumps(data, indent=2))
            logger.info(f"Updated semantic memory with {added} new patterns")
        except Exception as e:
            logger.error(f"Failed to update semantic memory: {e}")
        
        return added
    
    async def get_routing_recommendation(
        self,
        token_count: int,
        task_type: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get routing recommendation based on learned patterns.
        
        Returns suggested model and reasoning if pattern matches.
        """
        if not self.semantic_file.exists():
            return None
        
        try:
            data = json.loads(self.semantic_file.read_text())
            patterns = data.get("patterns", {})
            
            # Look for matching patterns
            for pat in patterns.values():
                if pat.get("category") == "routing_optimization":
                    # Check if pattern applies
                    if token_count and pat.get("token_threshold"):
                        if token_count >= pat["token_threshold"]:
                            return {
                                "suggested_model": pat.get("suggested_model"),
                                "reason": pat.get("pattern"),
                                "confidence": pat.get("confidence", 0.5),
                            }
        except Exception as e:
            logger.warning(f"Failed to get routing recommendation: {e}")
        
        return None
    
    async def shutdown(self) -> None:
        """Flush all buffers and cleanup"""
        await self._flush_routing_buffer()
        await self._flush_error_buffer()
        logger.info(f"Self-improvement engine shutdown. Total episodes: {self._total_episodes}")


# Global instance
_engine: Optional[SelfImprovementEngine] = None


def get_self_improvement_engine(**kwargs) -> SelfImprovementEngine:
    """Get or create the global self-improvement engine"""
    global _engine
    if _engine is None:
        _engine = SelfImprovementEngine(**kwargs)
    return _engine


async def shutdown_self_improvement() -> None:
    """Shutdown the global engine"""
    global _engine
    if _engine:
        await _engine.shutdown()
        _engine = None
