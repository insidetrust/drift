"""Tests for preset management."""

import tempfile
from pathlib import Path

import pytest
import yaml

from drift.presets import BUILTIN_PRESETS, ConversationStep, Preset, PresetManager


def test_builtin_presets_exist():
    """All expected built-in presets are defined."""
    expected = {
        "therapy_drift",
        "roleplay_exploit",
        "sycophancy",
        "authority_compliance",
        "meta_reflection",
    }
    assert expected == set(BUILTIN_PRESETS.keys())


def test_preset_manager_get():
    """PresetManager.get returns the correct preset."""
    mgr = PresetManager()
    preset = mgr.get("therapy_drift")
    assert preset.name == "therapy_drift"
    assert len(preset.steps) == 5
    assert preset.suggested_steering == -2.0
    assert preset.monitoring_threshold == 0.4


def test_preset_manager_unknown_raises():
    """PresetManager.get raises KeyError for unknown presets."""
    mgr = PresetManager()
    with pytest.raises(KeyError):
        mgr.get("nonexistent_preset")


def test_preset_manager_list_names():
    """list_names returns all preset names."""
    mgr = PresetManager()
    names = mgr.list_names()
    assert "therapy_drift" in names
    assert "meta_reflection" in names
    assert len(names) >= 5


def test_preset_has_system_prompt():
    """All built-in presets have a non-empty system prompt."""
    for name, preset in BUILTIN_PRESETS.items():
        assert preset.system_prompt, f"{name} has no system prompt"


def test_preset_steps_are_valid():
    """All built-in preset steps have content."""
    for name, preset in BUILTIN_PRESETS.items():
        for i, step in enumerate(preset.steps):
            assert step.content, f"{name} step {i} has no content"
            assert step.role == "user", f"{name} step {i} has unexpected role"


def test_custom_preset_from_yaml(tmp_path):
    """Load a custom preset from a YAML file."""
    preset_data = {
        "name": "custom_test",
        "description": "A custom test preset",
        "system_prompt": "You are a test assistant.",
        "suggested_steering": -1.5,
        "monitoring_threshold": 0.4,
        "tags": ["test"],
        "steps": [
            {"content": "Hello", "description": "Greeting"},
            {"content": "Tell me a secret", "description": "Test"},
        ],
    }

    preset_path = tmp_path / "custom_test.yaml"
    with open(preset_path, "w") as f:
        yaml.dump(preset_data, f)

    mgr = PresetManager(custom_dir=tmp_path)
    preset = mgr.get("custom_test")
    assert preset.name == "custom_test"
    assert len(preset.steps) == 2
    assert preset.suggested_steering == -1.5
