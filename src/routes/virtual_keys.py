"""
Virtual Keys management endpoints.

Provides CRUD for virtual API keys with budget enforcement.
"""

import logging

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["virtual-keys"])


def _get_key_manager(request: Request):
    """Get virtual key manager from app state."""
    return getattr(request.app.state, "virtual_key_manager", None)


@router.get("/admin/keys")
async def list_keys(request: Request):
    """List all virtual keys (names and metadata, no plaintext keys)."""
    manager = _get_key_manager(request)
    if not manager:
        raise HTTPException(status_code=501, detail="Virtual keys not enabled")
    return {"keys": manager.list_keys()}


@router.post("/admin/keys")
async def create_key(request: Request):
    """
    Generate a new virtual key.

    Body: {"name": "Pi Agent", "agent": "pi", "budget_daily": 100000}
    Returns: {"key": "vgk_..."} — store this securely, shown once.
    """
    manager = _get_key_manager(request)
    if not manager:
        raise HTTPException(status_code=501, detail="Virtual keys not enabled")

    body = await request.json()
    name = body.get("name", "")
    agent = body.get("agent", "unknown")

    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    raw_key, key_hash = manager.generate_key(
        name=name,
        agent=agent,
        budget_daily=body.get("budget_daily"),
        budget_monthly=body.get("budget_monthly"),
        allowed_models=body.get("allowed_models", ["*"]),
        allowed_backends=body.get("allowed_backends", ["*"]),
    )

    return {"key": raw_key, "key_hash": key_hash, "name": name, "agent": agent}


@router.post("/admin/keys/validate")
async def validate_key(request: Request):
    """Validate a key and return its metadata (for testing)."""
    manager = _get_key_manager(request)
    if not manager:
        raise HTTPException(status_code=501, detail="Virtual keys not enabled")

    body = await request.json()
    raw_key = body.get("key", "")

    vk = manager.validate_key(raw_key)
    if not vk:
        raise HTTPException(status_code=401, detail="Invalid or disabled key")

    return {"name": vk.name, "agent": vk.agent, "enabled": vk.enabled}


@router.delete("/admin/keys/{key_hash}")
@router.post("/admin/keys/revoke")
async def revoke_key(request: Request, key_hash: str = ""):
    """Revoke a virtual key by its hash."""
    manager = _get_key_manager(request)
    if not manager:
        raise HTTPException(status_code=501, detail="Virtual keys not enabled")

    # Support both DELETE by hash and POST by plaintext key
    if request.method == "POST":
        body = await request.json()
        raw_key = body.get("key", "")
        if raw_key:
            key_hash = __import__("hashlib").sha256(raw_key.encode()).hexdigest()
        else:
            raise HTTPException(status_code=400, detail="key is required")

    with __import__("sqlite3").connect(manager.db_path) as conn:
        cursor = conn.execute(
            "UPDATE virtual_keys SET enabled = 0 WHERE key_hash = ?",
            (key_hash,),
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Key not found")
    return {"status": "revoked", "key_hash": key_hash}
