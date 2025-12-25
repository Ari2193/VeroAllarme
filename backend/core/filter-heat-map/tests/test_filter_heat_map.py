"""
Integration-style tests for FilterHeatMap routing logic.
All files live under core/filter-heat-map/tests so imports remain local.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# Load the heat-map package dynamically because the folder name contains a hyphen.
ROOT = Path(__file__).resolve().parents[4]
PKG_DIR = ROOT / "backend" / "core" / "filter-heat-map"
PKG_NAME = "filter_heat_map_testpkg"


def _load_filter_class():
    """Dynamically load FilterHeatMap with a package context so relative imports work."""
    pkg = types.ModuleType(PKG_NAME)
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules[PKG_NAME] = pkg

    spec = importlib.util.spec_from_file_location(
        f"{PKG_NAME}.filter",
        PKG_DIR / "filter.py",
        submodule_search_locations=[str(PKG_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.FilterHeatMap


@pytest.fixture(scope="module")
def FilterHeatMap():  # noqa: N802 (pytest fixture naming)
    return _load_filter_class()


def _make_mask(shape=(32, 32), center=(16, 16), radius=5) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    y, x = center
    y0, y1 = max(0, y - radius), min(shape[0], y + radius)
    x0, x1 = max(0, x - radius), min(shape[1], x + radius)
    mask[y0:y1, x0:x1] = 255
    return mask


def test_cold_path_routes_to_stage_5(FilterHeatMap, tmp_path):
    storage = tmp_path / "cold"
    storage.mkdir()
    filter_instance = FilterHeatMap(
        camera_id="cam-cold",
        storage_path=str(storage),
        frame_shape=(32, 32),
        decay_factor=1.0,
        anomaly_threshold=0.4,  # below the default 0.5 cold score when history is short
    )

    mask = _make_mask()
    result = filter_instance.process_event(mask)

    assert result["flagged"] is True
    assert result["heat_zone"] == "cold"
    assert result["next_stage"] == 5
    assert result["anomaly_score"] >= 0.4
    assert any(storage.glob("heatmap_cam-cold.pkl"))


def test_hot_path_routes_to_stage_4(FilterHeatMap, tmp_path):
    storage = tmp_path / "hot"
    storage.mkdir()
    filter_instance = FilterHeatMap(
        camera_id="cam-hot",
        storage_path=str(storage),
        frame_shape=(32, 32),
        decay_factor=1.0,  # keep history strong
        anomaly_threshold=0.6,
    )

    mask = _make_mask(center=(10, 10))

    # Build enough history so anomaly scoring uses the statistical path (>=10 events)
    result = None
    for _ in range(12):
        result = filter_instance.process_event(mask)

    assert result is not None
    assert result["flagged"] is False
    assert result["heat_zone"] == "hot"
    assert result["next_stage"] == 4
    assert result["anomaly_score"] < 0.6
    assert any(storage.glob("heatmap_cam-hot.pkl"))
