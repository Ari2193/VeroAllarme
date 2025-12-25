"""
Tests for the Alert Processing Pipeline (Stages 3, 4, 5 integration).
"""

import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import importlib.util

# Handle hyphenated imports
backend_path = Path(__file__).parent.parent

# Setup filter-heat-map import
filter_heat_map_path = backend_path / "core" / "filter-heat-map"
spec = importlib.util.spec_from_file_location(
    "filter_heat_map",
    filter_heat_map_path / "__init__.py"
)
filter_heat_map_module = importlib.util.module_from_spec(spec)
sys.modules["filter_heat_map"] = filter_heat_map_module
spec.loader.exec_module(filter_heat_map_module)

# Setup compare-events import
compare_events_path = backend_path / "core" / "compare-events"
spec2 = importlib.util.spec_from_file_location(
    "compare_events",
    compare_events_path / "__init__.py"
)
compare_events_module = importlib.util.module_from_spec(spec2)
sys.modules["compare_events"] = compare_events_module
spec2.loader.exec_module(compare_events_module)

sys.path.insert(0, str(backend_path))

from core.pipeline import AlertProcessingPipeline, create_pipeline


class MockHeatMapFilter:
    """Mock Stage 3 heat map filter."""
    
    def process_event(self, motion_mask, original_image, save_overlay=False):
        # Simulate a cold zone detection
        return {
            "anomaly_score": 0.95,
            "flagged": True,
            "heat_zone": "cold",
            "next_stage": 5,
            "stats": {"quiet_zones": 3, "active_zones": 1},
        }


class MockAnomalyDetector:
    """Mock Stage 4 anomaly detector."""
    
    def __init__(self, decision="FILTER"):
        self.decision = decision
    
    def detect(self, camera_id, images, motion_boxes, top_k=10):
        return {
            "event_sim": 0.88,
            "event_support": 5,
            "decision": self.decision,
            "neighbors": [{"event_id": f"e{i}", "similarity": 0.85} for i in range(3)],
        }


class MockYOLODetector:
    """Mock Stage 5 YOLO detector."""
    
    def detect(self, image, confidence=0.6):
        return {
            "detections": [
                {"class": "person", "confidence": 0.95, "bbox": [10, 20, 100, 150]},
            ]
        }


def create_dummy_images(h=480, w=640, c=3):
    """Create dummy RGB images."""
    return [np.random.randint(0, 256, (h, w, c), dtype=np.uint8) for _ in range(3)]


def create_dummy_mask(h=480, w=640):
    """Create dummy motion mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[100:200, 100:200] = 255
    return mask


@pytest.fixture
def pipeline_components():
    """Create mock pipeline components."""
    return {
        "heatmap_filter": MockHeatMapFilter(),
        "anomaly_detector": MockAnomalyDetector(decision="FILTER"),
        "yolo_detector": MockYOLODetector(),
    }


class TestAlertProcessingPipeline:
    """Test alert processing pipeline."""
    
    def test_pipeline_creation(self, pipeline_components):
        """Test that pipeline is created correctly."""
        pipeline = create_pipeline(
            heatmap_filter=pipeline_components["heatmap_filter"],
            anomaly_detector=pipeline_components["anomaly_detector"],
            yolo_detector=pipeline_components["yolo_detector"],
        )
        assert pipeline is not None
        assert isinstance(pipeline, AlertProcessingPipeline)
    
    def test_alert_process_stage_4_filter_decision(self, pipeline_components):
        """Test Stage 4 FILTER decision routes to Stage 5."""
        pipeline = create_pipeline(**pipeline_components)
        
        images = create_dummy_images()
        mask = create_dummy_mask()
        
        result = pipeline.process_alert(
            camera_id="camera_1",
            alert_id="alert_001",
            images=images,
            motion_mask=mask,
            motion_boxes=[(100, 100, 200, 200)],
        )
        
        assert result["final_decision"] == "ESCALATE"
        assert result["stage_4_result"]["decision"] == "FILTER"
        assert "Stage 4" in result["explanation"]
    
    def test_alert_process_stage_4_drop_decision(self, pipeline_components):
        """Test Stage 4 DROP decision drops alert."""
        pipeline_components["anomaly_detector"] = MockAnomalyDetector(decision="DROP")
        pipeline = create_pipeline(**pipeline_components)
        
        images = create_dummy_images()
        mask = create_dummy_mask()
        
        result = pipeline.process_alert(
            camera_id="camera_1",
            alert_id="alert_002",
            images=images,
            motion_mask=mask,
            motion_boxes=[(100, 100, 200, 200)],
        )
        
        assert result["final_decision"] == "DROP"
        assert result["stage_4_result"]["decision"] == "DROP"
        assert "Low similarity" in result["explanation"]
    
    def test_alert_process_includes_stage_3_data(self, pipeline_components):
        """Test that Stage 3 heat map data is included."""
        pipeline = create_pipeline(**pipeline_components)
        
        images = create_dummy_images()
        mask = create_dummy_mask()
        
        result = pipeline.process_alert(
            camera_id="camera_1",
            alert_id="alert_003",
            images=images,
            motion_mask=mask,
            motion_boxes=[(100, 100, 200, 200)],
        )
        
        assert result["stage_3_result"] is not None
        assert result["stage_3_result"]["heat_zone"] == "cold"
        assert result["stage_3_result"]["next_stage"] == 5
        assert result["stage_3_result"]["anomaly_score"] == 0.95
    
    def test_alert_includes_metadata(self, pipeline_components):
        """Test that alert result includes required metadata."""
        pipeline = create_pipeline(**pipeline_components)
        
        images = create_dummy_images()
        mask = create_dummy_mask()
        
        result = pipeline.process_alert(
            camera_id="camera_2",
            alert_id="alert_004",
            images=images,
            motion_mask=mask,
            motion_boxes=[],
        )
        
        assert "alert_id" in result
        assert "camera_id" in result
        assert "timestamp" in result
        assert "final_decision" in result
        assert "explanation" in result
        assert result["camera_id"] == "camera_2"
        assert result["alert_id"] == "alert_004"
    
    def test_alert_stage_4_pass_with_cold_zone(self, pipeline_components):
        """Test Stage 4 PASS with Stage 3 cold zone → ESCALATE."""
        pipeline_components["anomaly_detector"] = MockAnomalyDetector(decision="PASS")
        pipeline = create_pipeline(**pipeline_components)
        
        images = create_dummy_images()
        mask = create_dummy_mask()
        
        result = pipeline.process_alert(
            camera_id="camera_1",
            alert_id="alert_005",
            images=images,
            motion_mask=mask,
            motion_boxes=[(100, 100, 200, 200)],
        )
        
        # Stage 3 returns heat_zone='cold' → next_stage=5
        # Stage 4 PASS → Use Stage 3 hint → ESCALATE
        assert result["final_decision"] == "ESCALATE"
        assert "Stage 3 hint: heat_zone='cold'" in result["explanation"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
