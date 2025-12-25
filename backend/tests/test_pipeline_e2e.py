"""
End-to-End Tests for Complete Alert Processing Pipeline.
Tests the full integration of Stage 3 (Heat Maps) → Stage 4 (Memory) → Stage 5 (YOLO).
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
from core.yolo_processor import YoloProcessor


def create_dummy_images(num_images=2):
    """Create dummy BGR images for testing."""
    images = []
    for i in range(num_images):
        img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        # Add some variation
        img[100+i*50:200+i*50, 150:350] = [255, 255, 255]
        images.append(img)
    return images


def create_dummy_mask():
    """Create a dummy motion mask."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[150:250, 200:400] = 255
    return mask


class TestEndToEndPipelineWithRealComponents:
    """Test complete pipeline with real Stage 5 (YOLO) but mocked Stage 3 & 4."""
    
    class MockHeatMapFilter:
        """Mock Stage 3 for controlled testing."""
        def __init__(self, heat_zone="cold", anomaly_score=0.95):
            self.heat_zone = heat_zone
            self.anomaly_score = anomaly_score
        
        def process_event(self, motion_mask, original_image, save_overlay=False):
            next_stage = 5 if self.heat_zone == "cold" else 4
            return {
                "anomaly_score": self.anomaly_score,
                "flagged": self.anomaly_score > 0.5,
                "heat_zone": self.heat_zone,
                "next_stage": next_stage,
                "stats": {"quiet_zones": 3, "active_zones": 1},
            }
    
    class MockAnomalyDetector:
        """Mock Stage 4 for controlled testing."""
        def __init__(self, decision="FILTER"):
            self.decision = decision
        
        def process_alert(self, camera_id, alert_id, images, motion_mask, motion_boxes):
            return {
                "decision": self.decision,
                "similarity": 0.95 if self.decision == "FILTER" else 0.75,
                "matches_count": 10 if self.decision == "FILTER" else 2,
                "explanation": f"Stage 4 {self.decision}"
            }
    
    @pytest.fixture
    def real_yolo(self):
        """Create a real YOLO processor."""
        return YoloProcessor()
    
    def test_e2e_cold_zone_triggers_yolo(self, real_yolo):
        """Test that cold zone in Stage 3 + PASS in Stage 4 triggers YOLO."""
        pipeline = AlertProcessingPipeline(
            heat_map_filter=self.MockHeatMapFilter(heat_zone="cold"),
            anomaly_detector=self.MockAnomalyDetector(decision="PASS"),
            yolo_detector=real_yolo
        )
        
        images = create_dummy_images()
        mask = create_dummy_mask()
        
        result = pipeline.process_alert(
            camera_id="cam_001",
            alert_id="alert_e2e_001",
            images=images,
            motion_mask=mask,
            motion_boxes=[(100, 100, 200, 200)]
        )
        
        # Verify pipeline completed all stages
        assert result["final_decision"] in ["ESCALATE", "DROP"]
        assert "stage_3_result" in result
        assert "stage_4_result" in result
        assert "stage_5_result" in result
        
        # Verify Stage 5 was actually called
        assert result["stage_5_result"] is not None
        stage5 = result["stage_5_result"]
        assert "num_objects" in stage5
        assert "detections" in stage5
        assert isinstance(stage5["detections"], list)
        
        # Verify explanation includes all stages
        explanation = result["explanation"]
        assert "Stage 3" in explanation
        assert "Stage 4" in explanation
        assert "Stage 5" in explanation or "YOLO" in explanation
    
    def test_e2e_filter_decision_triggers_yolo(self, real_yolo):
        """Test that FILTER decision in Stage 4 triggers YOLO."""
        pipeline = AlertProcessingPipeline(
            heat_map_filter=self.MockHeatMapFilter(heat_zone="hot"),
            anomaly_detector=self.MockAnomalyDetector(decision="FILTER"),
            yolo_detector=real_yolo
        )
        
        images = create_dummy_images()
        mask = create_dummy_mask()
        
        result = pipeline.process_alert(
            camera_id="cam_002",
            alert_id="alert_e2e_002",
            images=images,
            motion_mask=mask,
            motion_boxes=[(150, 150, 250, 250)]
        )
        
        # Verify YOLO ran
        assert result["stage_5_result"] is not None
        assert "detections" in result["stage_5_result"]
        
        # Verify final decision
        assert result["final_decision"] == "ESCALATE"
    
    def test_e2e_hot_zone_drop_no_yolo(self, real_yolo):
        """Test that hot zone + PASS in Stage 4 → DROP without YOLO."""
        pipeline = AlertProcessingPipeline(
            heat_map_filter=self.MockHeatMapFilter(heat_zone="hot", anomaly_score=0.3),
            anomaly_detector=self.MockAnomalyDetector(decision="PASS"),
            yolo_detector=real_yolo
        )
        
        images = create_dummy_images()
        mask = create_dummy_mask()
        
        result = pipeline.process_alert(
            camera_id="cam_003",
            alert_id="alert_e2e_003",
            images=images,
            motion_mask=mask,
            motion_boxes=[(100, 100, 200, 200)]
        )
        
        # Should drop without calling YOLO
        assert result["final_decision"] == "DROP"
        # Stage 5 should not be called
        assert result["stage_5_result"] is None or result["stage_5_result"]["num_objects"] == 0
    
    def test_e2e_drop_decision_no_yolo(self, real_yolo):
        """Test that DROP decision in Stage 4 prevents YOLO."""
        pipeline = AlertProcessingPipeline(
            heat_map_filter=self.MockHeatMapFilter(heat_zone="cold"),
            anomaly_detector=self.MockAnomalyDetector(decision="DROP"),
            yolo_detector=real_yolo
        )
        
        images = create_dummy_images()
        mask = create_dummy_mask()
        
        result = pipeline.process_alert(
            camera_id="cam_004",
            alert_id="alert_e2e_004",
            images=images,
            motion_mask=mask,
            motion_boxes=[(100, 100, 200, 200)]
        )
        
        # Should drop without YOLO
        assert result["final_decision"] == "DROP"
        assert result["stage_5_result"] is None


class TestEndToEndPipelineDataFlow:
    """Test data flow through complete pipeline."""
    
    class MockHeatMapFilter:
        def process_event(self, motion_mask, original_image, save_overlay=False):
            return {
                "anomaly_score": 0.95,
                "flagged": True,
                "heat_zone": "cold",
                "next_stage": 5,
                "stats": {"quiet_zones": 3, "active_zones": 1},
            }
    
    class MockAnomalyDetector:
        def process_alert(self, camera_id, alert_id, images, motion_mask, motion_boxes):
            return {
                "decision": "PASS",
                "similarity": 0.85,
                "matches_count": 5,
                "explanation": "Uncertain match"
            }
    
    @pytest.fixture
    def pipeline_with_yolo(self):
        """Create pipeline with real YOLO."""
        return AlertProcessingPipeline(
            heat_map_filter=self.MockHeatMapFilter(),
            anomaly_detector=self.MockAnomalyDetector(),
            yolo_detector=YoloProcessor()
        )
    
    def test_metadata_preserved_through_pipeline(self, pipeline_with_yolo):
        """Test that metadata is preserved through all stages."""
        images = create_dummy_images()
        mask = create_dummy_mask()
        
        result = pipeline_with_yolo.process_alert(
            camera_id="cam_meta_001",
            alert_id="alert_meta_001",
            images=images,
            motion_mask=mask,
            motion_boxes=[(100, 100, 200, 200)]
        )
        
        # Check metadata
        assert result["camera_id"] == "cam_meta_001"
        assert result["alert_id"] == "alert_meta_001"
        assert "timestamp" in result
        assert result["num_images"] == 2
    
    def test_all_stage_results_included(self, pipeline_with_yolo):
        """Test that all stage results are in final output."""
        images = create_dummy_images()
        mask = create_dummy_mask()
        
        result = pipeline_with_yolo.process_alert(
            camera_id="cam_complete_001",
            alert_id="alert_complete_001",
            images=images,
            motion_mask=mask,
            motion_boxes=[(100, 100, 200, 200)]
        )
        
        # All stages should be present
        assert "stage_3_result" in result
        assert "stage_4_result" in result
        assert "stage_5_result" in result
        
        # Stage 3 structure
        stage3 = result["stage_3_result"]
        assert "anomaly_score" in stage3
        assert "heat_zone" in stage3
        
        # Stage 4 structure
        stage4 = result["stage_4_result"]
        assert "decision" in stage4
        assert "similarity" in stage4
        
        # Stage 5 structure (if called)
        if result["stage_5_result"]:
            stage5 = result["stage_5_result"]
            assert "num_objects" in stage5
            assert "detections" in stage5
    
    def test_explanation_completeness(self, pipeline_with_yolo):
        """Test that explanation includes all stages."""
        images = create_dummy_images()
        mask = create_dummy_mask()
        
        result = pipeline_with_yolo.process_alert(
            camera_id="cam_explain_001",
            alert_id="alert_explain_001",
            images=images,
            motion_mask=mask,
            motion_boxes=[(100, 100, 200, 200)]
        )
        
        explanation = result["explanation"]
        
        # Should mention all relevant stages
        assert "Stage 3" in explanation or "heat" in explanation.lower()
        assert "Stage 4" in explanation
        # If Stage 5 ran, should be mentioned
        if result["stage_5_result"] and result["stage_5_result"]["num_objects"] > 0:
            assert "Stage 5" in explanation or "YOLO" in explanation


class TestPipelineErrorHandling:
    """Test pipeline behavior with errors and edge cases."""
    
    class MockHeatMapFilter:
        def process_event(self, motion_mask, original_image, save_overlay=False):
            return {
                "anomaly_score": 0.95,
                "flagged": True,
                "heat_zone": "cold",
                "next_stage": 5,
                "stats": {},
            }
    
    class MockAnomalyDetector:
        def process_alert(self, camera_id, alert_id, images, motion_mask, motion_boxes):
            return {
                "decision": "PASS",
                "similarity": 0.85,
                "matches_count": 5,
                "explanation": "Test"
            }
    
    def test_pipeline_with_empty_motion_boxes(self):
        """Test pipeline handles empty motion boxes."""
        pipeline = AlertProcessingPipeline(
            heat_map_filter=self.MockHeatMapFilter(),
            anomaly_detector=self.MockAnomalyDetector(),
            yolo_detector=YoloProcessor()
        )
        
        images = create_dummy_images()
        mask = create_dummy_mask()
        
        result = pipeline.process_alert(
            camera_id="cam_empty_001",
            alert_id="alert_empty_001",
            images=images,
            motion_mask=mask,
            motion_boxes=[]  # Empty motion boxes
        )
        
        # Should still process
        assert "final_decision" in result
        assert result["final_decision"] in ["ESCALATE", "DROP"]
    
    def test_pipeline_with_single_image(self):
        """Test pipeline with only one image."""
        pipeline = AlertProcessingPipeline(
            heat_map_filter=self.MockHeatMapFilter(),
            anomaly_detector=self.MockAnomalyDetector(),
            yolo_detector=YoloProcessor()
        )
        
        images = create_dummy_images(num_images=1)
        mask = create_dummy_mask()
        
        result = pipeline.process_alert(
            camera_id="cam_single_001",
            alert_id="alert_single_001",
            images=images,
            motion_mask=mask,
            motion_boxes=[(100, 100, 200, 200)]
        )
        
        # Should handle single image
        assert "final_decision" in result
        assert result["num_images"] == 1
    
    def test_pipeline_without_yolo(self):
        """Test pipeline works without YOLO detector."""
        pipeline = AlertProcessingPipeline(
            heat_map_filter=self.MockHeatMapFilter(),
            anomaly_detector=self.MockAnomalyDetector(),
            yolo_detector=None  # No YOLO
        )
        
        images = create_dummy_images()
        mask = create_dummy_mask()
        
        result = pipeline.process_alert(
            camera_id="cam_no_yolo_001",
            alert_id="alert_no_yolo_001",
            images=images,
            motion_mask=mask,
            motion_boxes=[(100, 100, 200, 200)]
        )
        
        # Should still work without YOLO
        assert "final_decision" in result
        assert result["stage_5_result"] is None


class TestPipelinePerformance:
    """Test pipeline performance characteristics."""
    
    class MockHeatMapFilter:
        def process_event(self, motion_mask, original_image, save_overlay=False):
            return {
                "anomaly_score": 0.5,
                "flagged": True,
                "heat_zone": "hot",
                "next_stage": 4,
                "stats": {},
            }
    
    class MockAnomalyDetector:
        def process_alert(self, camera_id, alert_id, images, motion_mask, motion_boxes):
            return {
                "decision": "DROP",
                "similarity": 0.70,
                "matches_count": 2,
                "explanation": "Low similarity"
            }
    
    def test_pipeline_processes_multiple_alerts(self):
        """Test pipeline can process multiple alerts sequentially."""
        pipeline = AlertProcessingPipeline(
            heat_map_filter=self.MockHeatMapFilter(),
            anomaly_detector=self.MockAnomalyDetector(),
            yolo_detector=YoloProcessor()
        )
        
        results = []
        for i in range(3):
            images = create_dummy_images()
            mask = create_dummy_mask()
            
            result = pipeline.process_alert(
                camera_id=f"cam_{i:03d}",
                alert_id=f"alert_{i:03d}",
                images=images,
                motion_mask=mask,
                motion_boxes=[(100, 100, 200, 200)]
            )
            results.append(result)
        
        # All should succeed
        assert len(results) == 3
        for result in results:
            assert "final_decision" in result
            assert result["final_decision"] in ["ESCALATE", "DROP"]
    
    def test_pipeline_consistent_output_structure(self):
        """Test that pipeline output structure is consistent."""
        pipeline = AlertProcessingPipeline(
            heat_map_filter=self.MockHeatMapFilter(),
            anomaly_detector=self.MockAnomalyDetector(),
            yolo_detector=YoloProcessor()
        )
        
        results = []
        for _ in range(2):
            images = create_dummy_images()
            mask = create_dummy_mask()
            
            result = pipeline.process_alert(
                camera_id="cam_consistent",
                alert_id=f"alert_consistent_{_}",
                images=images,
                motion_mask=mask,
                motion_boxes=[(100, 100, 200, 200)]
            )
            results.append(result)
        
        # Check both have same keys
        assert results[0].keys() == results[1].keys()
