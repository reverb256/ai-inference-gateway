"""
Cost tracking plugin for the AI Inference Gateway.

Tracks token usage per model, agent key, and backend.
Provides endpoint for querying aggregated costs.
"""

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Single token usage record."""

    timestamp: float
    model: str
    agent_key: str
    backend: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class AgentKey:
    """Virtual API key with metadata."""

    name: str
    budget: Optional[float] = None  # None = unlimited
    allowed_models: List[str] = field(default_factory=lambda: ["*"])


class CostTracker:
    """
    Track token usage and costs across models and agents.

    Uses SQLite for persistence (single-file, zero-config).
    """

    def __init__(self, db_path: str = "/var/cache/ai-inference/token_usage.db"):
        self.db_path = db_path
        self._ensure_dir()
        self._init_db()

    def _ensure_dir(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    model TEXT NOT NULL,
                    agent_key TEXT NOT NULL DEFAULT '',
                    backend TEXT NOT NULL DEFAULT '',
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON token_usage(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model ON token_usage(model)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent ON token_usage(agent_key)
            """)

    async def record(
        self,
        model: str,
        agent_key: str = "",
        backend: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record a token usage event."""
        total = input_tokens + output_tokens
        if total == 0:
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO token_usage (timestamp, model, agent_key, backend, input_tokens, output_tokens, total_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (time.time(), model, agent_key, backend, input_tokens, output_tokens, total),
            )
        logger.debug(
            f"Cost tracked: model={model}, in={input_tokens}, out={output_tokens}, agent={agent_key}"
        )

    async def get_summary(
        self,
        period: str = "daily",
        group_by: str = "model",
        agent_key: Optional[str] = None,
    ) -> Dict:
        """
        Get token usage summary.

        Args:
            period: 'hourly', 'daily', 'weekly', 'monthly', or 'all'
            group_by: 'model', 'agent', 'backend'
            agent_key: Filter to specific agent key

        Returns:
            Dict with usage summary
        """
        # Calculate time window
        now = time.time()
        period_seconds = {
            "hourly": 3600,
            "daily": 86400,
            "weekly": 604800,
            "monthly": 2592000,
            "all": 0,
        }
        since = now - period_seconds.get(period, 86400)

        group_col = {"model": "model", "agent": "agent_key", "backend": "backend"}.get(
            group_by, "model"
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = f"""
                SELECT
                    {group_col} as group_key,
                    COUNT(*) as request_count,
                    SUM(input_tokens) as total_input,
                    SUM(output_tokens) as total_output,
                    SUM(total_tokens) as total_tokens
                FROM token_usage
                WHERE timestamp >= ?
                {"AND agent_key = ?" if agent_key else ""}
                GROUP BY {group_col}
                ORDER BY total_tokens DESC
            """

            params: list = [since]
            if agent_key:
                params.append(agent_key)

            rows = conn.execute(query, params).fetchall()

            return {
                "period": period,
                "group_by": group_by,
                "since": since,
                "groups": [
                    {
                        "key": row["group_key"],
                        "requests": row["request_count"],
                        "input_tokens": row["total_input"],
                        "output_tokens": row["total_output"],
                        "total_tokens": row["total_tokens"],
                    }
                    for row in rows
                ],
            }

    async def get_agent_spend(
        self,
        agent_key: str,
        period: str = "daily",
    ) -> int:
        """Get total tokens consumed by an agent in a period."""
        now = time.time()
        period_seconds = {"hourly": 3600, "daily": 86400, "weekly": 604800, "monthly": 2592000, "all": 0}
        since = now - period_seconds.get(period, 86400)

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(total_tokens), 0) as total
                FROM token_usage
                WHERE timestamp >= ? AND agent_key = ?
                """,
                (since, agent_key),
            ).fetchone()
            return row[0] if row else 0
