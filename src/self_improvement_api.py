"""
Self-Improvement API Endpoints for AI Inference Gateway.

Adds REST endpoints for:
- Pattern extraction and review
- Episodic memory querying
- Feedback collection
- Learning status
- Integration with Hermes Agent
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

try:
    from .self_improvement import (
        SelfImprovementEngine,
        get_self_improvement_engine,
        EpisodeOutcome,
        RoutingEpisode,
        ErrorEpisode,
        PatternCandidate,
    )
    from .hermes_integration import (
        HermesBridge,
        get_hermes_bridge,
    )
    SELF_IMPROVEMENT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Self-improvement module not available: {e}")
    SELF_IMPROVEMENT_AVAILABLE = False
    SelfImprovementEngine = None
    get_self_improvement_engine = None
    EpisodeOutcome = None
    HermesBridge = None
    get_hermes_bridge = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/self-improvement", tags=["self-improvement"])


# Request/Response Models
class RoutingLogRequest(BaseModel):
    """Request to log a routing decision"""
    model_requested: str
    model_routed: str
    routing_reason: str
    token_count: int
    task_type: str
    latency_ms: float
    backend_used: str = "unknown"
    retry_count: int = 0
    fallback_triggered: bool = False
    error: Optional[str] = None


class ErrorLogRequest(BaseModel):
    """Request to log an error episode"""
    error_type: str
    error_message: str
    context: Dict[str, Any] = Field(default_factory=dict)
    resolution: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Request to record feedback"""
    episode_id: str
    rating: int = Field(ge=1, le=10, description="Rating from 1-10")
    comments: Optional[str] = None


class PatternExtractionRequest(BaseModel):
    """Request to extract patterns from episodic memory"""
    min_occurrences: int = Field(default=3, ge=1, le=20)
    auto_update: bool = Field(default=True, description="Update semantic memory automatically")


class SkillGenerationRequest(BaseModel):
    """Request to generate a Hermes skill from patterns"""
    skill_name: str
    description: str
    patterns: List[str]


class EpisodeListResponse(BaseModel):
    """Response containing episode list"""
    date: str
    total: int
    episodes: List[Dict[str, Any]]


class PatternListResponse(BaseModel):
    """Response containing extracted patterns"""
    patterns: List[Dict[str, Any]]
    total: int
    extraction_date: str


class LearningStatusResponse(BaseModel):
    """Response containing learning system status"""
    enabled: bool
    total_episodes: int
    patterns_extracted: int
    semantic_patterns_count: int
    hermes_integration: bool
    last_updated: Optional[str] = None


# Endpoints
@router.get("/status", response_model=LearningStatusResponse)
async def get_learning_status():
    """Get the current status of the self-improvement system"""
    if not SELF_IMPROVEMENT_AVAILABLE:
        return LearningStatusResponse(
            enabled=False,
            total_episodes=0,
            patterns_extracted=0,
            semantic_patterns_count=0,
            hermes_integration=False,
        )
    
    engine = get_self_improvement_engine()
    bridge = get_hermes_bridge()
    
    # Count semantic patterns
    semantic_file = Path("/run/ai-inference/memory/semantic-patterns.json")
    semantic_count = 0
    if semantic_file.exists():
        try:
            import json
            data = json.loads(semantic_file.read_text())
            semantic_count = len(data.get("patterns", {}))
        except Exception:
            pass
    
    return LearningStatusResponse(
        enabled=engine.enabled,
        total_episodes=engine._total_episodes,
        patterns_extracted=engine._patterns_extracted,
        semantic_patterns_count=semantic_count,
        hermes_integration=bridge.enabled if bridge else False,
        last_updated=datetime.now().isoformat(),
    )


@router.post("/log/routing")
async def log_routing_decision(request: RoutingLogRequest):
    """Log a routing decision episode for learning"""
    if not SELF_IMPROVEMENT_AVAILABLE:
        raise HTTPException(status_code=501, detail="Self-improvement not available")
    
    engine = get_self_improvement_engine()
    bridge = get_hermes_bridge()
    
    # Log to engine
    episode_id = await engine.log_routing_decision(
        model_requested=request.model_requested,
        model_routed=request.model_routed,
        routing_reason=request.routing_reason,
        token_count=request.token_count,
        task_type=request.task_type,
        latency_ms=request.latency_ms,
        backend_used=request.backend_used,
        retry_count=request.retry_count,
        fallback_triggered=request.fallback_triggered,
        error=request.error,
    )
    
    # Sync to Hermes if available
    if bridge:
        await bridge.sync_gateway_to_hermes(
            routing_decision=request.model_dump(),
            outcome="success" if not request.error else "failure",
        )
    
    return {"episode_id": episode_id, "status": "logged"}


@router.post("/log/error")
async def log_error_episode(request: ErrorLogRequest):
    """Log an error episode for self-correction"""
    if not SELF_IMPROVEMENT_AVAILABLE:
        raise HTTPException(status_code=501, detail="Self-improvement not available")
    
    engine = get_self_improvement_engine()
    
    episode_id = await engine.log_error(
        error_type=request.error_type,
        error_message=request.error_message,
        context=request.context,
        resolution=request.resolution,
    )
    
    return {"episode_id": episode_id, "status": "logged"}


@router.post("/feedback")
async def record_feedback(request: FeedbackRequest):
    """Record user feedback for an episode"""
    if not SELF_IMPROVEMENT_AVAILABLE:
        raise HTTPException(status_code=501, detail="Self-improvement not available")
    
    engine = get_self_improvement_engine()
    success = await engine.record_feedback(
        episode_id=request.episode_id,
        rating=request.rating,
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    return {"status": "recorded", "episode_id": request.episode_id}


@router.get("/episodes/{date}")
async def get_episodes(
    date: str,
    limit: int = Query(default=50, ge=1, le=500),
    outcome: Optional[str] = Query(default=None, description="Filter by outcome"),
) -> EpisodeListResponse:
    """Get episodes for a specific date"""
    episodic_dir = Path("/run/ai-inference/memory/episodic")
    episode_file = episodic_dir / f"{date}-gateway.json"
    
    if not episode_file.exists():
        raise HTTPException(status_code=404, detail="No episodes found for this date")
    
    try:
        import json
        data = json.loads(episode_file.read_text())
        episodes = data.get("episodes", [])
        
        # Filter by outcome if specified
        if outcome:
            episodes = [e for e in episodes if e.get("outcome") == outcome]
        
        # Limit results
        episodes = episodes[:limit]
        
        return EpisodeListResponse(
            date=date,
            total=len(episodes),
            episodes=episodes,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read episodes: {e}")


@router.post("/patterns/extract", response_model=PatternListResponse)
async def extract_patterns(request: PatternExtractionRequest):
    """Extract patterns from accumulated episodic memory"""
    if not SELF_IMPROVEMENT_AVAILABLE:
        raise HTTPException(status_code=501, detail="Self-improvement not available")
    
    engine = get_self_improvement_engine()
    
    # Extract patterns
    patterns = await engine.extract_patterns(min_occurrences=request.min_occurrences)
    
    # Update semantic memory if requested
    added = 0
    if request.auto_update:
        added = await engine.update_semantic_memory(patterns)
    
    return PatternListResponse(
        patterns=[p.to_dict() for p in patterns],
        total=len(patterns),
        extraction_date=datetime.now().isoformat(),
    )


@router.get("/patterns")
async def get_semantic_patterns(
    category: Optional[str] = Query(default=None),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
):
    """Get learned semantic patterns"""
    semantic_file = Path("/run/ai-inference/memory/semantic-patterns.json")
    
    if not semantic_file.exists():
        return {"patterns": {}, "total": 0}
    
    try:
        import json
        data = json.loads(semantic_file.read_text())
        patterns = data.get("patterns", {})
        
        # Filter by category
        if category:
            patterns = {
                k: v for k, v in patterns.items()
                if v.get("category") == category
            }
        
        # Filter by confidence
        if min_confidence > 0:
            patterns = {
                k: v for k, v in patterns.items()
                if v.get("confidence", 0) >= min_confidence
            }
        
        return {
            "patterns": patterns,
            "total": len(patterns),
            "categories": list(set(p.get("category") for p in patterns.values())),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read patterns: {e}")


@router.post("/hermes/skill")
async def create_hermes_skill(request: SkillGenerationRequest):
    """Create a new Hermes skill from learned patterns"""
    if not SELF_IMPROVEMENT_AVAILABLE:
        raise HTTPException(status_code=501, detail="Self-improvement not available")
    
    bridge = get_hermes_bridge()
    if not bridge or not bridge.enabled:
        raise HTTPException(status_code=501, detail="Hermes integration not available")
    
    try:
        skill_file = await bridge.create_shared_skill(
            skill_name=request.skill_name,
            description=request.description,
            patterns=request.patterns,
        )
        
        return {
            "status": "created",
            "skill_file": str(skill_file),
            "skill_name": request.skill_name,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create skill: {e}")


@router.get("/hermes/context")
async def get_hermes_context():
    """Get current Hermes context for routing decisions"""
    if not SELF_IMPROVEMENT_AVAILABLE:
        raise HTTPException(status_code=501, detail="Self-improvement not available")
    
    bridge = get_hermes_bridge()
    if not bridge or not bridge.enabled:
        raise HTTPException(status_code=501, detail="Hermes integration not available")
    
    context = await bridge.get_hermes_context()
    return context


@router.post("/patterns/sync")
async def sync_cross_system_patterns():
    """Extract and sync patterns across Gateway and Hermes"""
    if not SELF_IMPROVEMENT_AVAILABLE:
        raise HTTPException(status_code=501, detail="Self-improvement not available")
    
    bridge = get_hermes_bridge()
    if not bridge or not bridge.enabled:
        raise HTTPException(status_code=501, detail="Hermes integration not available")
    
    patterns = await bridge.extract_cross_system_patterns()
    
    return {
        "status": "synced",
        "patterns_found": len(patterns),
        "patterns": patterns[:10],  # Return first 10
    }


@router.post("/flush")
async def flush_memory_buffers():
    """Manually flush all memory buffers"""
    if not SELF_IMPROVEMENT_AVAILABLE:
        raise HTTPException(status_code=501, detail="Self-improvement not available")
    
    engine = get_self_improvement_engine()
    await engine._flush_routing_buffer()
    await engine._flush_error_buffer()
    
    return {"status": "flushed"}


def create_self_improvement_router(**config) -> APIRouter:
    """Create and configure the self-improvement router"""
    # Initialize engine with config
    if SELF_IMPROVEMENT_AVAILABLE:
        # Extract relevant parameters for each component
        engine_config = {
            k: v for k, v in config.items()
            if k in {'enabled', 'auto_extract_patterns', 'min_confidence_for_update', 'memory_base'}
        }
        bridge_config = {
            k: v for k, v in config.items()
            if k in {'enabled', 'hermes_memory', 'cluster_memory'}
        }

        get_self_improvement_engine(**engine_config)
        get_hermes_bridge(**bridge_config)
        logger.info("Self-improvement system initialized")

    return router
