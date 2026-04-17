"""
Hermes Agent Integration for AI Inference Gateway.

Provides bidirectional learning between:
- AI Gateway: Routing decisions, error patterns, performance metrics
- Hermes Agent: Task execution, skill usage, cluster operations

Shared memory location: /var/lib/hermes/memory/
Cluster memory location: /run/ai-inference/memory/
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)


# Dual memory paths
HERMES_MEMORY = Path("/var/lib/hermes/memory")
# Use /run for self-improvement memory (writable by ai-inference user)
CLUSTER_MEMORY_BASE = Path("/run/ai-inference")
CLUSTER_MEMORY = CLUSTER_MEMORY_BASE / "memory"
EPISODIC_DIR = CLUSTER_MEMORY / "episodic"


class HermesBridge:
    """
    Bridge between AI Gateway and Hermes Agent.
    
    Enables:
    - Gateway to share routing patterns with Hermes
    - Hermes to share skill outcomes with Gateway
    - Combined pattern extraction from both systems
    - Unified feedback collection
    """
    
    def __init__(
        self,
        hermes_memory: Path = HERMES_MEMORY,
        cluster_memory: Path = CLUSTER_MEMORY,
        enabled: bool = True,
    ):
        self.hermes_memory = hermes_memory
        self.cluster_memory = cluster_memory
        self.enabled = enabled

        # Ensure cluster memory directories exist (we have write access here)
        self.cluster_memory.mkdir(parents=True, exist_ok=True)
        EPISODIC_DIR.mkdir(parents=True, exist_ok=True)

        # Try to create hermes memory, but don't fail if we can't
        # (systemd should create this directory with proper permissions)
        try:
            self.hermes_memory.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            logger.warning(f"Cannot create hermes memory directory: {e}")
            logger.info(f"Hermes integration will use cluster memory only")
            # Use cluster memory as fallback
            self.hermes_memory = self.cluster_memory / "hermes"
    
    async def sync_gateway_to_hermes(
        self,
        routing_decision: Dict[str, Any],
        outcome: str,
    ) -> None:
        """
        Sync gateway routing decision to Hermes memory.
        
        This allows Hermes to learn:
        - Which models work best for which tasks
        - Current cluster state and load
        - Recent error patterns to avoid
        """
        if not self.enabled:
            return
        
        timestamp = datetime.now().isoformat()
        
        # Create shared memory entry
        entry = {
            "source": "ai-gateway",
            "timestamp": timestamp,
            "routing_decision": routing_decision,
            "outcome": outcome,
        }
        
        # Write to Hermes-accessible location
        gateway_log = self.hermes_memory / "gateway_events.jsonl"
        try:
            with open(gateway_log, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write to Hermes memory: {e}")
    
    async def sync_hermes_to_cluster(
        self,
        task: str,
        skill_used: str,
        outcome: str,
        lessons: Optional[List[str]] = None,
    ) -> None:
        """
        Sync Hermes task execution to cluster memory.
        
        This enables:
        - Pattern extraction across all cluster operations
        - Skill effectiveness tracking
        - Cross-system learning
        """
        if not self.enabled:
            return
        
        timestamp = datetime.now().isoformat()
        episode_id = f"hermes-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create episodic memory entry
        episode = {
            "id": episode_id,
            "timestamp": timestamp,
            "source": "hermes-agent",
            "skill": skill_used,
            "task": task,
            "outcome": outcome,
            "lessons": lessons or [],
            "related_patterns": [],
        }
        
        # Write to cluster episodic memory
        today = datetime.now().strftime("%Y-%m-%d")
        episode_file = EPISODIC_DIR / f"{today}-hermes.json"
        
        try:
            if episode_file.exists():
                data = json.loads(episode_file.read_text())
            else:
                data = {"date": today, "episodes": []}
            
            data["episodes"].append(episode)
            episode_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to write to cluster memory: {e}")
    
    async def get_hermes_context(self) -> Dict[str, Any]:
        """
        Get current Hermes context for routing decisions.
        
        Returns:
        - Active skills and their effectiveness
        - Recent task outcomes
        - Current cluster state
        """
        if not self.enabled:
            return {}
        
        context = {
            "active_skills": [],
            "recent_outcomes": [],
            "cluster_state": {},
        }
        
        # Read recent Hermes events
        hermes_log = self.hermes_memory / "events.jsonl"
        if hermes_log.exists():
            try:
                recent_events = []
                with open(hermes_log, "r") as f:
                    for line in f:
                        if line.strip():
                            recent_events.append(json.loads(line))
                
                # Get last 10 events
                context["recent_outcomes"] = recent_events[-10:]
            except Exception as e:
                logger.warning(f"Failed to read Hermes context: {e}")
        
        return context
    
    async def record_skill_feedback(
        self,
        skill_name: str,
        rating: int,  # 1-10
        comments: Optional[str] = None,
    ) -> None:
        """
        Record feedback for a Hermes skill.
        
        This feedback is used for:
        - Skill effectiveness ranking
        - Pattern confidence updates
        - Automatic skill improvement
        """
        if not self.enabled:
            return
        
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "skill": skill_name,
            "rating": rating,
            "comments": comments,
        }
        
        # Write to feedback log
        feedback_log = self.hermes_memory / "skill_feedback.jsonl"
        try:
            with open(feedback_log, "a") as f:
                f.write(json.dumps(feedback_entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to record skill feedback: {e}")
    
    async def extract_cross_system_patterns(self) -> List[Dict[str, Any]]:
        """
        Extract patterns that span both Gateway and Hermes.
        
        Examples:
        - "Hermes deployment tasks work best with qwen3.5-4b model"
        - "Gateway errors spike after Hermes runs nixos-rebuild"
        - "High latency on cluster status queries"
        """
        if not self.enabled:
            return []
        
        patterns = []
        
        # Look for correlations between Gateway and Hermes events
        gateway_events = []
        hermes_events = []
        
        gateway_log = self.hermes_memory / "gateway_events.jsonl"
        if gateway_log.exists():
            try:
                with open(gateway_log, "r") as f:
                    for line in f:
                        gateway_events.append(json.loads(line))
            except Exception:
                pass
        
        hermes_log = self.hermes_memory / "events.jsonl"
        if hermes_log.exists():
            try:
                with open(hermes_log, "r") as f:
                    for line in f:
                        hermes_events.append(json.loads(line))
            except Exception:
                pass
        
        # Analyze for patterns
        # (Pattern extraction logic would go here)
        
        return patterns
    
    async def create_shared_skill(
        self,
        skill_name: str,
        description: str,
        patterns: List[str],
    ) -> Path:
        """
        Create a new Hermes skill from learned patterns.
        
        This enables automatic skill generation from:
        - Repeated task sequences
        - Successful resolution patterns
        - User-validated workflows
        """
        if not self.enabled:
            raise RuntimeError("Hermes bridge is not enabled")
        
        skill_content = f"""---
name: {skill_name}
description: {description}
version: 1.0.0
source: auto-generated
metadata:
  hermes:
    tags: [auto-generated, learned]
  generated_at: {datetime.now().isoformat()}
---

# {skill_name}

{description}

## Learned Patterns

{chr(10).join(f"- {p}" for p in patterns)}

## Usage

This skill was automatically generated from observed patterns
in the AI Gateway and Hermes Agent operations.

## Related Patterns

See cluster memory: /run/ai-inference/memory/semantic-patterns.json
"""
        
        # Write to Hermes skills directory
        skills_dir = self.hermes_memory.parent / "skills" / "auto-generated"
        skills_dir.mkdir(parents=True, exist_ok=True)
        
        skill_file = skills_dir / f"{skill_name}.md"
        skill_file.write_text(skill_content)
        
        logger.info(f"Created auto-generated skill: {skill_file}")
        return skill_file


# Global bridge instance
_bridge: Optional[HermesBridge] = None


def get_hermes_bridge(**kwargs) -> HermesBridge:
    """Get or create the global Hermes bridge"""
    global _bridge
    if _bridge is None:
        _bridge = HermesBridge(**kwargs)
    return _bridge
