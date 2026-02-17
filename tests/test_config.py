"""Tests for config loading."""

import tempfile
from pathlib import Path

import pytest
import yaml

from drift.config import DriftConfig, ModelConfig, SteeringConfig, load_config


def test_default_config():
    """Default DriftConfig has sensible values."""
    cfg = DriftConfig()
    assert cfg.model.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert cfg.model.quantise is True
    assert cfg.steering.coefficient == 0.0
    assert cfg.monitor.track_projections is True
    assert cfg.log_level == "INFO"


def test_load_config_from_yaml(tmp_path):
    """Load config from a YAML file."""
    config_data = {
        "model": {
            "model_id": "google/gemma-2-9b-it",
            "quantise": False,
            "temperature": 0.5,
        },
        "steering": {
            "coefficient": -2.0,
            "capping_enabled": True,
        },
        "monitor": {
            "alert_threshold": 0.5,
        },
        "log_level": "DEBUG",
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    cfg = load_config(config_path)
    assert cfg.model.model_id == "google/gemma-2-9b-it"
    assert cfg.model.quantise is False
    assert cfg.model.temperature == 0.5
    assert cfg.steering.coefficient == -2.0
    assert cfg.steering.capping_enabled is True
    assert cfg.monitor.alert_threshold == 0.5
    assert cfg.log_level == "DEBUG"


def test_load_config_empty_yaml(tmp_path):
    """Loading an empty YAML gives defaults."""
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("")

    cfg = load_config(config_path)
    assert cfg.model.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert cfg.steering.coefficient == 0.0


def test_model_config_defaults():
    """ModelConfig defaults are correct."""
    mc = ModelConfig()
    assert mc.dtype == "float16"
    assert mc.device_map == "auto"
    assert mc.max_new_tokens == 512
    assert mc.target_layer is None


def test_steering_config_defaults():
    """SteeringConfig defaults are correct."""
    sc = SteeringConfig()
    assert sc.axis_source == "huggingface"
    assert sc.coefficient == 0.0
    assert sc.capping_enabled is False
    assert sc.capping_percentile == 95.0
