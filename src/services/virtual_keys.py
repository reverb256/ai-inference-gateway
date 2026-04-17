"""
Virtual Keys and Budget Enforcement.

Provides scoped API keys for each agent (Pi, Claude Code, Hermes, etc.)
with optional token budget caps per time period.
"""

import hashlib
import logging
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class VirtualKey:
    """A scoped API key with metadata and budget."""

    key_hash: str  # SHA256 of the actual key (never stored plaintext)
    name: str
    agent: str  # pi | claude-code | hermes | crush | droid | opencode
    budget_daily: Optional[int] = None  # Max tokens per day (None = unlimited)
    budget_monthly: Optional[int] = None  # Max tokens per month (None = unlimited)
    allowed_models: List[str] = field(default_factory=lambda: ["*"])
    allowed_backends: List[str] = field(default_factory=lambda: ["*"])
    enabled: bool = True
    created_at: float = 0.0
    last_used_at: float = 0.0

    def check_model_allowed(self, model: str) -> bool:
        if "*" in self.allowed_models:
            return True
        return any(m in model for m in self.allowed_models)

    def check_backend_allowed(self, backend: str) -> bool:
        if "*" in self.allowed_backends:
            return True
        return backend in self.allowed_backends


class VirtualKeyManager:
    """
    Manages virtual API keys with budget enforcement.

    Uses SQLite for persistence. Keys are stored as SHA256 hashes.
    """

    def __init__(self, db_path: str = "/var/cache/ai-inference/virtual_keys.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS virtual_keys (
                    key_hash TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    budget_daily INTEGER,
                    budget_monthly INTEGER,
                    allowed_models TEXT DEFAULT '["*"]',
                    allowed_backends TEXT DEFAULT '["*"]',
                    enabled INTEGER DEFAULT 1,
                    created_at REAL NOT NULL,
                    last_used_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS key_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    key_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    tokens INTEGER NOT NULL,
                    FOREIGN KEY (key_hash) REFERENCES virtual_keys(key_hash)
                )
            """)

    def generate_key(self, name: str, agent: str, **kwargs) -> tuple:
        """Generate a new virtual key. Returns (plaintext_key, key_hash)."""
        raw_key = f"vgk_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO virtual_keys (key_hash, name, agent, budget_daily, budget_monthly,
                    allowed_models, allowed_backends, enabled, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    key_hash,
                    name,
                    agent,
                    kwargs.get("budget_daily"),
                    kwargs.get("budget_monthly"),
                    json.dumps(kwargs.get("allowed_models", ["*"])),
                    json.dumps(kwargs.get("allowed_backends", ["*"])),
                    time.time(),
                ),
            )

        logger.info(f"Generated virtual key for {name} (agent={agent})")
        return raw_key, key_hash
        return raw_key

    def validate_key(self, raw_key: str) -> Optional[VirtualKey]:
        """Validate a key and return its metadata. Returns None if invalid."""
        if not raw_key or not raw_key.startswith("vgk_"):
            return None

        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT * FROM virtual_keys WHERE key_hash = ? AND enabled = 1
                """,
                (key_hash,),
            ).fetchone()

        if not row:
            return None

        return VirtualKey(
            key_hash=row["key_hash"],
            name=row["name"],
            agent=row["agent"],
            budget_daily=row["budget_daily"],
            budget_monthly=row["budget_monthly"],
            allowed_models=json.loads(row["allowed_models"]),
            allowed_backends=json.loads(row["allowed_backends"]),
            enabled=bool(row["enabled"]),
            created_at=row["created_at"],
            last_used_at=row["last_used_at"],
        )

    async def check_budget(self, key: VirtualKey, period: str = "daily") -> bool:
        """Check if key is within budget. Returns True if OK, False if over."""
        if not key.budget_daily and not key.budget_monthly:
            return True

        budget = {"daily": key.budget_daily, "monthly": key.budget_monthly}.get(period)
        if budget is None:
            return True

        period_seconds = {"daily": 86400, "monthly": 2592000}
        since = time.time() - period_seconds.get(period, 86400)

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(tokens), 0) as total
                FROM key_usage
                WHERE timestamp >= ? AND key_hash = ?
                """,
                (since, key.key_hash),
            ).fetchone()
            spent = row[0] if row else 0

        return spent < budget

    async def record_usage(
        self, key: VirtualKey, model: str, backend: str, tokens: int
    ) -> None:
        """Record token usage for a key."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO key_usage (timestamp, key_hash, model, backend, tokens)
                VALUES (?, ?, ?, ?, ?)
                """,
                (time.time(), key.key_hash, model, backend, tokens),
            )
            conn.execute(
                "UPDATE virtual_keys SET last_used_at = ? WHERE key_hash = ?",
                (time.time(), key.key_hash),
            )

    def list_keys(self) -> List[Dict]:
        """List all virtual keys (without plaintext)."""

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT key_hash, name, agent, budget_daily, budget_monthly,
                       enabled, created_at, last_used_at
                FROM virtual_keys
                ORDER BY created_at DESC
                """
            ).fetchall()

        return [dict(r) for r in rows]


# Need json for allowed_models/backends parsing
import json  # noqa: E402
from typing import Dict  # noqa: E402
