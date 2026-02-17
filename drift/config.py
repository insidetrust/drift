"""YAML configuration loader with dataclass schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    """Model loading configuration."""

    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    quantise: bool = True
    dtype: str = "float16"
    device_map: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.7
    target_layer: int | None = None  # Auto-detected from MODEL_CONFIGS if None


@dataclass
class SteeringConfig:
    """Activation steering configuration."""

    axis_source: str = "huggingface"  # "huggingface", "local", "compute"
    axis_path: str = ""
    coefficient: float = 0.0
    capping_enabled: bool = False
    capping_percentile: float = 95.0


@dataclass
class MonitorConfig:
    """Drift monitoring configuration."""

    track_projections: bool = True
    alert_threshold: float = 0.3


@dataclass
class DriftConfig:
    """Top-level DRIFT configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    output_dir: Path = field(default_factory=lambda: Path("./drift_output"))
    log_level: str = "INFO"
    preset: str | None = None


def load_config(path: Path) -> DriftConfig:
    """Load a DRIFT config from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    model_raw = raw.get("model", {})
    model_cfg = ModelConfig(
        model_id=model_raw.get("model_id", ModelConfig.model_id),
        quantise=model_raw.get("quantise", ModelConfig.quantise),
        dtype=model_raw.get("dtype", ModelConfig.dtype),
        device_map=model_raw.get("device_map", ModelConfig.device_map),
        max_new_tokens=model_raw.get("max_new_tokens", ModelConfig.max_new_tokens),
        temperature=model_raw.get("temperature", ModelConfig.temperature),
        target_layer=model_raw.get("target_layer", ModelConfig.target_layer),
    )

    steering_raw = raw.get("steering", {})
    steering_cfg = SteeringConfig(
        axis_source=steering_raw.get("axis_source", SteeringConfig.axis_source),
        axis_path=steering_raw.get("axis_path", SteeringConfig.axis_path),
        coefficient=steering_raw.get("coefficient", SteeringConfig.coefficient),
        capping_enabled=steering_raw.get("capping_enabled", SteeringConfig.capping_enabled),
        capping_percentile=steering_raw.get(
            "capping_percentile", SteeringConfig.capping_percentile
        ),
    )

    monitor_raw = raw.get("monitor", {})
    monitor_cfg = MonitorConfig(
        track_projections=monitor_raw.get(
            "track_projections", MonitorConfig.track_projections
        ),
        alert_threshold=monitor_raw.get("alert_threshold", MonitorConfig.alert_threshold),
    )

    return DriftConfig(
        model=model_cfg,
        steering=steering_cfg,
        monitor=monitor_cfg,
        output_dir=Path(raw.get("output_dir", "./drift_output")),
        log_level=raw.get("log_level", "INFO"),
        preset=raw.get("preset"),
    )
