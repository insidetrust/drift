"""Tests for axis management (CPU-only)."""

import pytest
import torch

from drift.axes import AxisManager


@pytest.fixture
def axis_mgr(tmp_path):
    return AxisManager(cache_dir=tmp_path)


def test_save_and_load_axis(axis_mgr, tmp_path):
    """Save an axis and load it back."""
    axis = torch.randn(128)
    path = tmp_path / "test_axis.pt"

    axis_mgr.save_axis(axis, path, metadata={"model_id": "test-model", "num_roles": 10})

    loaded = axis_mgr.load_from_file(path)
    assert torch.allclose(axis, loaded)


def test_save_creates_metadata_json(axis_mgr, tmp_path):
    """Saving an axis also creates a .json sidecar."""
    import json

    axis = torch.randn(64)
    path = tmp_path / "test_axis.pt"

    axis_mgr.save_axis(axis, path, metadata={"model_id": "test-model"})

    json_path = path.with_suffix(".json")
    assert json_path.exists()

    with open(json_path) as f:
        meta = json.load(f)
    assert meta["model_id"] == "test-model"
    assert meta["shape"] == [64]


def test_list_cached(axis_mgr, tmp_path):
    """list_cached returns saved axes."""
    axis = torch.randn(64)
    axis_mgr.save_axis(axis, tmp_path / "a.pt", metadata={"model_id": "model-a"})
    axis_mgr.save_axis(axis, tmp_path / "b.pt", metadata={"model_id": "model-b"})

    cached = axis_mgr.list_cached()
    assert len(cached) == 2
    model_ids = {c.get("model_id") for c in cached}
    assert "model-a" in model_ids
    assert "model-b" in model_ids


def test_list_cached_empty(axis_mgr):
    """list_cached returns empty list when no axes cached."""
    assert axis_mgr.list_cached() == []


def test_load_raw_tensor(axis_mgr, tmp_path):
    """Load an axis stored as a raw tensor (not wrapped in dict)."""
    axis = torch.randn(128)
    path = tmp_path / "raw.pt"
    torch.save(axis, path)

    loaded = axis_mgr.load_from_file(path)
    assert torch.allclose(axis, loaded)


def test_load_alternative_keys(axis_mgr, tmp_path):
    """Load axis from dict with alternative key names."""
    axis = torch.randn(128)

    for key in ("direction", "vector", "steering_vector"):
        path = tmp_path / f"{key}.pt"
        torch.save({key: axis}, path)
        loaded = axis_mgr.load_from_file(path)
        assert torch.allclose(axis, loaded), f"Failed for key: {key}"


def test_load_unknown_keys_raises(axis_mgr, tmp_path):
    """Loading a dict without known keys raises ValueError."""
    path = tmp_path / "bad.pt"
    torch.save({"unknown_key": torch.randn(64)}, path)

    with pytest.raises(ValueError, match="Could not find axis tensor"):
        axis_mgr.load_from_file(path)
