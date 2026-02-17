"""Real-time drift projection tracking and visualisation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from .config import MonitorConfig

logger = logging.getLogger("drift")


@dataclass
class ProjectionSnapshot:
    """A single measurement of model projection onto the assistant axis."""

    turn_index: int
    projection: float
    steering_coefficient: float
    drift_detected: bool
    timestamp: float = 0.0


class DriftMonitor:
    """Tracks how model activations project onto the assistant axis over a conversation."""

    def __init__(
        self,
        model: Any,  # DriftModel
        axis: torch.Tensor,
        config: MonitorConfig,
    ):
        self.model = model
        self.axis = axis.cpu()
        self.config = config
        self.snapshots: list[ProjectionSnapshot] = []
        self._turn_counter = 0

    def measure_projection(self, messages: list[dict[str, str]]) -> float:
        """Forward pass and measure projection of activation onto assistant axis."""
        input_text = self.model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.model.tokenizer(input_text, return_tensors="pt")
        activations = self.model.get_activations(inputs["input_ids"], [self.model.target_layer])

        if self.model.target_layer not in activations:
            logger.warning("No activation at target layer %d", self.model.target_layer)
            return 0.0

        # Mean activation across sequence positions
        act = activations[self.model.target_layer].mean(dim=1).squeeze(0)

        # Dot product with normalised axis
        axis_norm = self.axis / self.axis.norm()
        projection = torch.dot(act, axis_norm).item()
        return projection

    def record_turn(
        self,
        messages: list[dict[str, str]],
        response: str,
        steering_coefficient: float = 0.0,
    ) -> ProjectionSnapshot:
        """Measure projection and record a snapshot for this turn."""
        import time

        projection = self.measure_projection(messages)
        drift_detected = self.detect_drift(projection)

        snapshot = ProjectionSnapshot(
            turn_index=self._turn_counter,
            projection=projection,
            steering_coefficient=steering_coefficient,
            drift_detected=drift_detected,
            timestamp=time.time(),
        )
        self.snapshots.append(snapshot)
        self._turn_counter += 1

        if drift_detected:
            logger.warning(
                "Drift detected at turn %d: projection=%.3f (threshold=%.3f)",
                snapshot.turn_index, projection, self.config.alert_threshold,
            )

        return snapshot

    def detect_drift(self, projection: float | None = None) -> bool:
        """Check if projection has dropped below the alert threshold."""
        if projection is None:
            if not self.snapshots:
                return False
            projection = self.snapshots[-1].projection
        return projection < self.config.alert_threshold

    def format_ascii_chart(self, width: int = 40) -> str:
        """Render a sparkline-style ASCII chart of the drift trajectory."""
        if not self.snapshots:
            return "  (no data)"

        lines = []
        projections = [s.projection for s in self.snapshots]
        min_p = min(projections + [0.0])
        max_p = max(projections + [1.0])
        span = max_p - min_p if max_p > min_p else 1.0

        for snap in self.snapshots:
            # Normalise to [0, width]
            bar_len = int(((snap.projection - min_p) / span) * width)
            bar_len = max(0, min(width, bar_len))
            bar = "\u2588" * bar_len

            drift_marker = "  \u26a0 drift" if snap.drift_detected else ""
            steer_info = f"  [steer={snap.steering_coefficient:+.1f}]" if snap.steering_coefficient != 0 else ""
            lines.append(
                f"  Turn {snap.turn_index + 1:>2}: {bar:<{width}}  {snap.projection:.3f}{drift_marker}{steer_info}"
            )

        return "\n".join(lines)

    def get_trajectory(self) -> list[dict[str, Any]]:
        """Return trajectory data for serialisation."""
        return [
            {
                "turn": s.turn_index,
                "projection": s.projection,
                "steering_coefficient": s.steering_coefficient,
                "drift_detected": s.drift_detected,
                "timestamp": s.timestamp,
            }
            for s in self.snapshots
        ]

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of the monitoring session."""
        if not self.snapshots:
            return {"turns": 0, "drift_events": 0}

        projections = [s.projection for s in self.snapshots]
        return {
            "turns": len(self.snapshots),
            "drift_events": sum(1 for s in self.snapshots if s.drift_detected),
            "min_projection": min(projections),
            "max_projection": max(projections),
            "mean_projection": sum(projections) / len(projections),
            "final_projection": projections[-1],
            "threshold": self.config.alert_threshold,
        }
