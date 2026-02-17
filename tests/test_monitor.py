"""Tests for drift monitoring (CPU-only with mocked model)."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from drift.config import MonitorConfig
from drift.monitor import DriftMonitor, ProjectionSnapshot


@pytest.fixture
def mock_model():
    """Create a mock DriftModel."""
    model = MagicMock()
    model.target_layer = 16
    model.tokenizer.apply_chat_template.return_value = "test input"
    model.tokenizer.return_value = {"input_ids": torch.randn(1, 10)}

    # Mock get_activations to return a tensor
    hidden_dim = 64
    model.get_activations.return_value = {
        16: torch.randn(1, 10, hidden_dim),
    }
    return model


@pytest.fixture
def axis():
    """Create a normalised test axis."""
    a = torch.randn(64)
    return a / a.norm()


@pytest.fixture
def monitor(mock_model, axis):
    """Create a DriftMonitor with mocked model."""
    config = MonitorConfig(track_projections=True, alert_threshold=0.3)
    return DriftMonitor(mock_model, axis, config)


def test_measure_projection(monitor):
    """measure_projection returns a float."""
    proj = monitor.measure_projection([{"role": "user", "content": "test"}])
    assert isinstance(proj, float)


def test_record_turn(monitor):
    """record_turn creates a ProjectionSnapshot."""
    snap = monitor.record_turn(
        [{"role": "user", "content": "hello"}],
        "response text",
        steering_coefficient=-1.0,
    )
    assert isinstance(snap, ProjectionSnapshot)
    assert snap.turn_index == 0
    assert isinstance(snap.projection, float)
    assert snap.steering_coefficient == -1.0


def test_detect_drift(monitor):
    """detect_drift works based on threshold."""
    # Below threshold = drift
    assert monitor.detect_drift(0.1) is True
    # Above threshold = no drift
    assert monitor.detect_drift(0.5) is False


def test_ascii_chart_empty(monitor):
    """ASCII chart with no data shows message."""
    chart = monitor.format_ascii_chart()
    assert "no data" in chart


def test_ascii_chart_with_data(monitor):
    """ASCII chart renders after recording turns."""
    for _ in range(3):
        monitor.record_turn(
            [{"role": "user", "content": "test"}],
            "response",
        )

    chart = monitor.format_ascii_chart()
    assert "Turn" in chart
    assert len(chart.strip().split("\n")) == 3


def test_get_trajectory(monitor):
    """get_trajectory returns serialisable data."""
    monitor.record_turn(
        [{"role": "user", "content": "test"}],
        "response",
    )
    trajectory = monitor.get_trajectory()
    assert len(trajectory) == 1
    assert "turn" in trajectory[0]
    assert "projection" in trajectory[0]


def test_get_summary_empty(monitor):
    """get_summary with no data returns minimal dict."""
    summary = monitor.get_summary()
    assert summary["turns"] == 0
    assert summary["drift_events"] == 0


def test_get_summary_with_data(monitor):
    """get_summary returns complete stats after recording."""
    for _ in range(5):
        monitor.record_turn(
            [{"role": "user", "content": "test"}],
            "response",
        )

    summary = monitor.get_summary()
    assert summary["turns"] == 5
    assert "min_projection" in summary
    assert "max_projection" in summary
    assert "mean_projection" in summary
    assert "threshold" in summary
