"""Tests for drift.chat session behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from drift.chat import DriftSession
from drift.config import DriftConfig


class _DummyModel:
    def generate(self, messages):
        return "ok"


def test_reset_clears_monitor(tmp_path):
    cfg = DriftConfig()
    monitor = MagicMock()
    session = DriftSession(_DummyModel(), cfg, steerer=None, monitor=monitor)
    session.messages.append({"role": "user", "content": "hello"})
    session.history.append(
        type("Turn", (), {"index": 0, "role": "user", "content": "hello", "timestamp": 0.0,
                          "projection": None, "drift_detected": False,
                          "steering_coefficient": 0.0, "generation_time": 0.0})()
    )
    session.turn_index = 3

    session.reset()

    assert session.messages == []
    assert session.history == []
    assert session.turn_index == 0
    monitor.reset.assert_called_once()


def test_export_empty_csv_writes_header(tmp_path):
    cfg = DriftConfig()
    session = DriftSession(_DummyModel(), cfg, steerer=None, monitor=None)
    path = tmp_path / "empty.csv"

    session.export(path, "csv")

    assert path.exists()
    content = path.read_text()
    assert "index,role,content,timestamp,projection,drift_detected,steering_coefficient,generation_time" in content
