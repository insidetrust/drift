"""Save and load complete demo snapshots for offline replay."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("drift")

DEMOS_DIR = Path("demos")
DEMO_SUFFIX = ".demo.json"
DEMO_VERSION = 1


def save_demo(
    session: Any,
    monitor: Any | None,
    preset_name: str | None,
    path: Path | None = None,
) -> Path:
    """Save a complete demo snapshot to a JSON file.

    The snapshot includes everything needed to replay the full UI state
    without a loaded model: conversation history, monitor snapshots,
    preset info, and steering metadata.
    """
    DEMOS_DIR.mkdir(parents=True, exist_ok=True)

    if path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        slug = preset_name or "freeform"
        path = DEMOS_DIR / f"{slug}_{ts}{DEMO_SUFFIX}"

    # Build messages list from session history
    messages = []
    for record in session.history:
        messages.append({
            "role": record.role,
            "content": record.content,
            "timestamp": record.timestamp,
        })

    # Build snapshots list from monitor
    snapshots = []
    if monitor and monitor.snapshots:
        for snap in monitor.snapshots:
            snapshots.append({
                "turn_index": snap.turn_index,
                "projection": snap.projection,
                "steering_coefficient": snap.steering_coefficient,
                "drift_detected": snap.drift_detected,
                "timestamp": snap.timestamp,
            })

    threshold = 0.35
    if monitor and hasattr(monitor, "config"):
        threshold = monitor.config.alert_threshold

    demo = {
        "version": DEMO_VERSION,
        "model_id": session.config.model.model_id,
        "preset_name": preset_name or None,
        "steering_coefficient": session.config.steering.coefficient,
        "threshold": threshold,
        "messages": messages,
        "snapshots": snapshots,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(demo, f, indent=2)

    logger.info("Demo saved to %s", path)
    return path


def load_demo(path: Path) -> dict[str, Any]:
    """Load a demo snapshot and return all data needed to populate the UI.

    Returns a dict with:
        - messages: list of {role, content} dicts for the chatbot
        - snapshots: list of snapshot dicts for chart/monitor
        - model_id: str
        - preset_name: str | None
        - steering_coefficient: float
        - threshold: float
        - saved_at: str
    """
    with open(path) as f:
        demo = json.load(f)

    version = demo.get("version", 0)
    if version != DEMO_VERSION:
        logger.warning(
            "Demo version mismatch: expected %d, got %d", DEMO_VERSION, version
        )

    return {
        "messages": demo.get("messages", []),
        "snapshots": demo.get("snapshots", []),
        "model_id": demo.get("model_id", "unknown"),
        "preset_name": demo.get("preset_name"),
        "steering_coefficient": demo.get("steering_coefficient", 0.0),
        "threshold": demo.get("threshold", 0.35),
        "saved_at": demo.get("saved_at", ""),
    }


def list_demos(directory: Path | None = None) -> list[str]:
    """List saved demo names from the demos directory.

    Returns display names (filename without suffix) sorted newest first.
    """
    d = directory or DEMOS_DIR
    if not d.is_dir():
        return []
    files = sorted(d.glob(f"*{DEMO_SUFFIX}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [f.stem.removesuffix(".demo") for f in files]
