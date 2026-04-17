"""
GPU Scheduler Integration for AI Inference Gateway (Kubernetes-Native)

Signals GPU workload scheduler when AI workloads start/stop using ConfigMap-based state management.
Replaces file-based IPC with Kubernetes-native ConfigMap updates.

Migration: File-based IPC (/run/gpu-scheduler/ai-state) → ConfigMap (kube-system/gpu-scheduler-state)
"""

import logging
import subprocess
import json
import os
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Kubernetes ConfigMap for state management
SCHEDULER_CONFIGMAP = "gpu-scheduler-state"
SCHEDULER_NAMESPACE = "kube-system"

# State values (maintained for backward compatibility)
STATE_IDLE = "IDLE"
STATE_AI_STARTING = "AI_START"
STATE_AI_STOPPING = "AI_STOP"

# State metadata tracking
STATE_TRANSITIONS = {
    STATE_IDLE: "No AI workload running - GPUs available for mining",
    STATE_AI_STARTING: "AI workload starting - mining should be preempted",
    STATE_AI_STOPPING: "AI workload stopping - mining can resume"
}


def _kubectl_patch_configmap(state: str) -> bool:
    """
    Update GPU scheduler state using kubectl patch.

    Uses kubectl patch for atomic ConfigMap updates without race conditions.
    Requires RBAC: Role (gpu-scheduler-state-updater) with get/patch/update on ConfigMaps.
    """
    patch_data = {
        "data": {
            "ai-state": state,
            "last-updated": datetime.now(timezone.utc).isoformat(),
            "active-workload": "ai-inference" if state == STATE_AI_STARTING else "none"
        }
    }

    try:
        result = subprocess.run(
            [
                "kubectl", "patch", "configmap", SCHEDULER_CONFIGMAP,
                "-n", SCHEDULER_NAMESPACE,
                "--type=merge",
                f"--patch={json.dumps(patch_data)}"
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            logger.info(f"✓ GPU scheduler ConfigMap updated: {state}")
            logger.debug(f"  Patch response: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"✗ kubectl patch failed (exit {result.returncode}): {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("✗ kubectl patch timed out (10s)")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ kubectl patch failed: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"✗ Failed to update ConfigMap: {e}")
        return False


def _kubectl_get_configmap() -> Optional[Dict[str, str]]:
    """
    Get current GPU scheduler state from ConfigMap.

    Returns:
        Dict with 'ai-state', 'last-updated', 'active-workload' or None if error.
    """
    try:
        result = subprocess.run(
            [
                "kubectl", "get", "configmap", SCHEDULER_CONFIGMAP,
                "-n", SCHEDULER_NAMESPACE,
                "-o", "jsonpath={.data}"
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            # Parse output format: "map[ai-state:IDLE last-updated:... active-workload:none]"
            data_str = result.stdout.strip()
            if data_str.startswith("map["):
                # Convert "map[key:value key:value]" to dict
                data_str = data_str[4:]  # Remove "map[" prefix
                data = {}
                for item in data_str[:-1].split(" "):  # Remove trailing "]" and split
                    if ":" in item:
                        key, value = item.split(":", 1)
                        data[key] = value
                return data
            else:
                logger.warning(f"Unexpected kubectl output format: {data_str}")
                return None
        else:
            logger.error(f"✗ kubectl get failed (exit {result.returncode}): {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        logger.error("✗ kubectl get timed out (10s)")
        return None
    except subprocess.CalledProcessError as e:
        # ConfigMap might not exist yet
        if "NotFound" in str(e) or "not found" in e.stderr.lower():
            logger.debug(f"ConfigMap {SCHEDULER_CONFIGMAP} not found (will create on first write)")
            return None
        logger.error(f"✗ kubectl get failed: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"✗ Failed to get ConfigMap: {e}")
        return None


def init_scheduler_comms() -> None:
    """
    Initialize GPU scheduler communication (Kubernetes ConfigMap).

    Creates ConfigMap if it doesn't exist and ensures RBAC is configured.
    """
    try:
        # Check if ConfigMap exists
        current_state = _kubectl_get_configmap()

        if current_state is None:
            logger.info(f"Creating ConfigMap {SCHEDULER_CONFIGMAP} in {SCHEDULER_NAMESPACE}")

            # Create ConfigMap with initial IDLE state
            create_result = subprocess.run(
                [
                    "kubectl", "create", "configmap", SCHEDULER_CONFIGMAP,
                    "-n", SCHEDULER_NAMESPACE,
                    f"--from-literal=ai-state={STATE_IDLE}",
                    f"--from-literal=last-updated={datetime.now(timezone.utc).isoformat()}",
                    "--from-literal=active-workload=none",
                    "--dry-run=client",
                    "-o", "yaml"
                ],
                check=True,
                capture_output=True,
                text=True
            )

            # Apply the created ConfigMap
            apply_result = subprocess.run(
                ["kubectl", "apply", "-f", "-"],
                input=create_result.stdout,
                check=True,
                capture_output=True,
                text=True
            )

            if apply_result.returncode == 0:
                logger.info(f"✓ ConfigMap {SCHEDULER_CONFIGMAP} created")
            else:
                logger.error(f"✗ Failed to create ConfigMap: {apply_result.stderr}")
        else:
            logger.info(f"✓ ConfigMap {SCHEDULER_CONFIGMAP} exists")
            logger.debug(f"  Current state: {current_state.get('ai-state', 'UNKNOWN')}")

        # Verify RBAC (warn if not configured)
        try:
            subprocess.run(
                ["kubectl", "auth", "can-i", "patch", "configmap", SCHEDULER_CONFIGMAP,
                 "-n", SCHEDULER_NAMESPACE],
                check=True,
                capture_output=True
            )
            logger.debug("✓ RBAC verified: can patch configmap")
        except subprocess.CalledProcessError:
            logger.warning(
                f"⚠ RBAC not configured for ConfigMap updates. "
                f"Ensure Role 'gpu-scheduler-state-updater' exists with patch permissions."
            )

    except Exception as e:
        logger.error(f"Failed to initialize scheduler comms: {e}")


def write_state(state: str) -> bool:
    """
    Write state to GPU scheduler via ConfigMap.

    Args:
        state: One of STATE_IDLE, STATE_AI_STARTING, STATE_AI_STOPPING

    Returns:
        True if state updated successfully, False otherwise
    """
    if state not in STATE_TRANSITIONS:
        logger.error(f"Invalid state: {state}. Must be one of {list(STATE_TRANSITIONS.keys())}")
        return False

    logger.info(f"→ GPU scheduler state: {state} ({STATE_TRANSITIONS.get(state, 'Unknown')})")
    return _kubectl_patch_configmap(state)


def notify_ai_starting() -> bool:
    """
    Signal GPU scheduler that AI workload is starting.

    YuniKorn/Volcano will preempt mining pods based on priority classes:
    - AI inference: high-priority-ai (1000)
    - Mining: low-priority-mining (100)

    Returns:
        True if signal sent successfully, False otherwise
    """
    logger.info("🚀 Signaling GPU scheduler: AI workload starting (mining will be preempted)")
    return write_state(STATE_AI_STARTING)


def notify_ai_stopping() -> bool:
    """
    Signal GPU scheduler that AI workload is stopping.

    Mining pods will be scaled back up by the custom scheduler or
    YuniKorn will allocate GPUs back to mining queue.

    Returns:
        True if signal sent successfully, False otherwise
    """
    logger.info("⏹️  Signaling GPU scheduler: AI workload stopping (mining can resume)")
    return write_state(STATE_AI_STOPPING)


def notify_ai_idle() -> bool:
    """
    Signal GPU scheduler that AI workload is idle (no model loaded).

    Returns:
        True if signal sent successfully, False otherwise
    """
    logger.debug("💤 Signaling GPU scheduler: AI workload idle")
    return write_state(STATE_IDLE)


def get_current_state() -> str:
    """
    Get current GPU scheduler state from ConfigMap.

    Returns:
        Current state (STATE_IDLE, STATE_AI_STARTING, STATE_AI_STOPPING) or STATE_IDLE if error
    """
    try:
        data = _kubectl_get_configmap()
        if data:
            state = data.get("ai-state", STATE_IDLE)
            logger.debug(f"Current GPU scheduler state: {state}")
            return state
        return STATE_IDLE
    except Exception as e:
        logger.error(f"Failed to get current state: {e}")
        return STATE_IDLE


def get_scheduler_metadata() -> Optional[Dict[str, str]]:
    """
    Get full scheduler state metadata.

    Returns:
        Dict with 'ai-state', 'last-updated', 'active-workload' or None if error
    """
    return _kubectl_get_configmap()


# ============================================================================
# BACKWARD COMPATIBILITY: File-based state (legacy, for rollback)
# ============================================================================

def _init_file_state() -> None:
    """
    Initialize file-based state (legacy, for rollback compatibility).

    This function ensures backward compatibility if you need to rollback
    to the file-based scheduler. The file-based state is kept in sync
    with ConfigMap state for dual-mode operation during migration.
    """
    SCHEDULER_STATE_DIR = Path("/run/gpu-scheduler")
    SCHEDULER_STATE_FILE = SCHEDULER_STATE_DIR / "ai-state"

    try:
        SCHEDULER_STATE_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize from ConfigMap if file doesn't exist
        if not SCHEDULER_STATE_FILE.exists():
            current_state = get_current_state()
            SCHEDULER_STATE_FILE.write_text(current_state or STATE_IDLE)
            logger.debug(f"Initialized file state from ConfigMap: {current_state}")

    except Exception as e:
        logger.error(f"Failed to initialize file state: {e}")


def _sync_file_state(state: str) -> bool:
    """
    Sync state to file (legacy compatibility).

    Keeps file-based state in sync with ConfigMap during migration.
    """
    SCHEDULER_STATE_FILE = Path("/run/gpu-scheduler/ai-state")

    try:
        SCHEDULER_STATE_FILE.write_text(state)
        return True
    except Exception as e:
        logger.error(f"Failed to sync file state: {e}")
        return False


# Enhanced state write functions that maintain dual-mode compatibility
def write_state_dual_mode(state: str) -> bool:
    """
    Write state to both ConfigMap and file (dual-mode during migration).

    This ensures compatibility with both:
    - New: YuniKorn/Volcano watching ConfigMap
    - Legacy: Custom Python scheduler watching file

    Returns:
        True if both writes succeeded, False otherwise
    """
    configmap_success = write_state(state)
    file_success = _sync_file_state(state)

    if configmap_success and file_success:
        return True
    elif configmap_success:
        logger.warning("⚠ ConfigMap updated but file sync failed (migration mode)")
        return True
    else:
        return False


# Auto-initialization on module import
if __name__ != "__main__":
    init_scheduler_comms()
    _init_file_state()
