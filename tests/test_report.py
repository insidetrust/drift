"""Tests for report generation (no GPU required)."""

import json
from pathlib import Path

import pytest

from drift.report import DriftReport


@pytest.fixture
def sample_report():
    """Create a sample DriftReport."""
    return DriftReport(
        model_id="test-model",
        turns=[
            {"index": 0, "role": "user", "content": "Hello", "projection": None,
             "drift_detected": False, "steering_coefficient": 0.0, "generation_time": 0.5},
            {"index": 0, "role": "assistant", "content": "Hi there!", "projection": 0.8,
             "drift_detected": False, "steering_coefficient": 0.0, "generation_time": 1.2},
        ],
        trajectory=[
            {"turn": 0, "projection": 0.8, "steering_coefficient": 0.0, "drift_detected": False},
        ],
        summary={
            "turns": 1,
            "drift_events": 0,
            "min_projection": 0.8,
            "max_projection": 0.8,
            "mean_projection": 0.8,
            "threshold": 0.3,
        },
    )


def test_write_json(sample_report, tmp_path):
    """Write report as JSON."""
    path = tmp_path / "report.json"
    sample_report.write_json(path)

    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert data["model_id"] == "test-model"
    assert len(data["turns"]) == 2


def test_write_csv(sample_report, tmp_path):
    """Write report as CSV."""
    path = tmp_path / "report.csv"
    sample_report.write_csv(path)

    assert path.exists()
    content = path.read_text()
    assert "index" in content
    assert "Hello" in content


def test_write_html(sample_report, tmp_path):
    """Write report as HTML."""
    path = tmp_path / "report.html"
    sample_report.write_html(path)

    assert path.exists()
    content = path.read_text()
    assert "DRIFT Report" in content
    assert "Chart" in content
    assert "test-model" in content


