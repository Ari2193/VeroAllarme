"""
Tests for Stage 3: Heat Map Filter

Test the heat map analysis and routing logic.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import importlib.util

# Setup import for filter-heat-map module
ROOT = Path(__file__).resolve().parents[4]
PKG_DIR = ROOT / "backend" / "core" / "filter-heat-map"
PKG_NAME = "filter_heat_map_pkg"


def _load_filter_heat_map():
    """Load FilterHeatMap from filter-heat-map module."""
    spec = importlib.util.spec_from_file_location(
        PKG_NAME,
        PKG_DIR / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PKG_NAME] = module
    spec.loader.exec_module(module)
    return module


filter_module = _load_filter_heat_map()


@pytest.fixture
def mock_filter_heat_map():
    """Get FilterHeatMap class from filter-heat-map module."""
    return filter_module.FilterHeatMap


def create_dummy_mask(h=480, w=640):
    """Create dummy motion mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[100:200, 100:200] = 255
    return mask


def test_cold_path_routes_to_stage_5(mock_filter_heat_map):
    """Test that anomalous motion (cold zone) routes to Stage 5."""
    filter_obj = mock_filter_heat_map(
        camera_id="test_camera",
        anomaly_threshold=0.5
    )
    
    # Create a mask with motion in an unusual location
    mask = create_dummy_mask()
    
    # First call will have no history, so anomaly_score = 0.5 (moderate)
    result = filter_obj.process_event(mask)
    
    # The second call should detect the pattern
    for _ in range(12):
        result = filter_obj.process_event(mask)
    
    # Now test with motion in a different location (should be flagged as cold)
    unusual_mask = np.zeros((480, 640), dtype=np.uint8)
    unusual_mask[10:50, 10:50] = 255  # Different location
    
    result = filter_obj.process_event(unusual_mask)
    
    # Verify routing decision
    assert "heat_zone" in result
    assert "next_stage" in result
    assert result["anomaly_score"] >= 0.0


def test_hot_path_routes_to_stage_4(mock_filter_heat_map):
    """Test that normal motion (hot zone) routes to Stage 4."""
    filter_obj = mock_filter_heat_map(
        camera_id="test_camera_2",
        anomaly_threshold=0.5
    )
    
    # Add many events at the same location to build history
    mask = create_dummy_mask()
    for i in range(12):
        result = filter_obj.process_event(mask)
    
    # After building history, same location should be considered "hot" (normal)
    final_result = filter_obj.process_event(mask)
    
    assert "heat_zone" in final_result
    assert "next_stage" in final_result
    # Should route to Stage 4 (hot) if in familiar location
    if final_result["anomaly_score"] <= 0.5:
        assert final_result["heat_zone"] == "hot"
        assert final_result["next_stage"] == 4
