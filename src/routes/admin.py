"""
Cost tracking and admin endpoints.

Provides token usage metrics, virtual key management,
and budget enforcement for the AI Inference Gateway.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["admin"])


def _get_tracker(request: Request):
    """Get cost tracker from app state."""
    return getattr(request.app.state, "cost_tracker", None)


@router.get("/admin/costs")
async def get_costs(
    request: Request,
    period: str = "daily",
    group_by: str = "model",
    agent_key: Optional[str] = None,
):
    """
    Get token usage summary.

    Args:
        period: hourly | daily | weekly | monthly | all
        group_by: model | agent | backend
        agent_key: Filter to specific agent
    """
    tracker = _get_tracker(request)
    if not tracker:
        raise HTTPException(status_code=501, detail="Cost tracking not enabled")

    return await tracker.get_summary(
        period=period,
        group_by=group_by,
        agent_key=agent_key,
    )


@router.get("/admin/costs/agent/{agent_key}")
async def get_agent_costs(
    request: Request,
    agent_key: str,
    period: str = "daily",
):
    """Get token usage for a specific agent key."""
    tracker = _get_tracker(request)
    if not tracker:
        raise HTTPException(status_code=501, detail="Cost tracking not enabled")

    spend = await tracker.get_agent_spend(agent_key=agent_key, period=period)
    return {"agent_key": agent_key, "period": period, "total_tokens": spend}


@router.post("/admin/costs/reset")
async def reset_costs(request: Request):
    """Reset all cost tracking data."""
    tracker = _get_tracker(request)
    if not tracker:
        raise HTTPException(status_code=501, detail="Cost tracking not enabled")

    import sqlite3
    with sqlite3.connect(tracker.db_path) as conn:
        conn.execute("DELETE FROM token_usage")
    return {"status": "reset"}
