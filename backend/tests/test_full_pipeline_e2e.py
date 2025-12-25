"""
End-to-End Test: Full Pipeline (Stages 1-5)

Tests the complete alert processing pipeline:
- Stage 1: Motion Detection
- Stage 2: Masked Region Filtering (optional)
- Stage 3: HeatMap Analysis
- Stage 4: Memory-based Anomaly Detection
- Stage 5: YOLO Object Detection

Each test generates annotated images showing the flow through all stages.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import cv2

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.motion_detection import MotionDetector
from backend.core.anomaly_detection import AnomalyDetector
from backend.core.pipeline import create_pipeline
from backend.core.pipeline_visualizer import PipelineVisualizer

# Try to import YOLO, but make it optional
try:
    from backend.core.yolo_processor import YoloProcessor
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YoloProcessor = None

# Import with hyphenated directory name
import importlib
filter_heat_map = importlib.import_module('backend.core.filter-heat-map.filter')
FilterHeatMap = filter_heat_map.FilterHeatMap


class TestFullPipelineE2E:
    """Test the complete pipeline from Stage 1 to Stage 5."""
    
    @pytest.fixture
    def motion_detector(self):
        """Create motion detector (Stage 1)."""
        return MotionDetector(threshold=25, min_area=500)
    
    @pytest.fixture
    def heatmap_filter(self):
        """Create heatmap filter (Stage 3)."""
        return FilterHeatMap(camera_id="test_camera", frame_shape=(480, 640))
    
    @pytest.fixture
    def anomaly_detector(self):
        """Create anomaly detector (Stage 4)."""
        return AnomalyDetector()
    
    @pytest.fixture
    def yolo_detector(self):
        """Create YOLO detector (Stage 5) - optional."""
        if not YOLO_AVAILABLE:
            pytest.skip("YOLO (ultralytics) not installed")
        try:
            return YoloProcessor(model_name="yolov8n.pt")
        except Exception as e:
            pytest.skip(f"YOLO not available: {e}")
    
    @pytest.fixture
    def visualizer(self):
        """Create visualizer."""
        output_dir = Path("data/test_outputs/full_pipeline")
        output_dir.mkdir(parents=True, exist_ok=True)
        return PipelineVisualizer(output_dir=str(output_dir))
    
    @pytest.fixture
    def sample_frames(self):
        """Generate three consecutive sample frames with motion."""
        # Frame 1: Background
        frame1 = np.ones((480, 640, 3), dtype=np.uint8) * 100
        
        # Frame 2: Object appears (motion)
        frame2 = frame1.copy()
        cv2.rectangle(frame2, (200, 150), (350, 300), (150, 150, 150), -1)
        
        # Frame 3: Object moved
        frame3 = frame1.copy()
        cv2.rectangle(frame3, (220, 170), (370, 320), (150, 150, 150), -1)
        
        return [frame1, frame2, frame3]
    
    def test_stage_1_motion_detection(self, motion_detector, sample_frames, visualizer):
        """Test Stage 1: Motion Detection."""
        # Save frames temporarily
        temp_dir = Path("data/test_outputs/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_paths = []
        for i, frame in enumerate(sample_frames):
            temp_path = temp_dir / f"frame_{i}.jpg"
            cv2.imwrite(str(temp_path), frame)
            temp_paths.append(str(temp_path))
        
        # Detect motion between frames
        result = motion_detector.detect_motion(temp_paths[:2])
        
        assert result.motion_detected, "Motion should be detected"
        assert len(result.motion_regions) > 0, "Should have motion regions"
        assert result.motion_mask is not None, "Should have motion mask"
        
        # Visualize Stage 1 result
        img = sample_frames[1].copy()
        for region in result.motion_regions:
            x1, y1, x2, y2 = region.to_bbox()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        lines = [
            "STAGE 1: Motion Detection",
            f"Motion Detected: {result.motion_detected}",
            f"Motion Regions: {len(result.motion_regions)}",
            f"Total Motion Area: {result.total_motion_area} pixels",
            f"Confidence: {result.confidence:.3f}"
        ]
        img = visualizer.add_text_overlay(img, lines, bg_color=(0, 50, 0))
        
        output_path = visualizer.output_dir / "test_stage1_motion_detection.jpg"
        cv2.imwrite(str(output_path), img)
        
        # Cleanup
        for temp_path in temp_paths:
            Path(temp_path).unlink()
        
        print(f"✓ Stage 1 visualization saved: {output_path}")
    
    def test_stage_1_to_3_integration(self, motion_detector, heatmap_filter, 
                                      sample_frames, visualizer):
        """Test Stage 1 + Stage 3 integration."""
        # Save frames temporarily
        temp_dir = Path("data/test_outputs/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_paths = [str(temp_dir / f"frame_{i}.jpg") for i in range(2)]
        for i, frame in enumerate(sample_frames[:2]):
            cv2.imwrite(temp_paths[i], frame)
        
        # Stage 1: Motion Detection
        motion_result = motion_detector.detect_motion(temp_paths)
        motion_mask = motion_result.motion_mask
        motion_boxes = [region.to_bbox() for region in motion_result.motion_regions]
        
        # Cleanup
        for temp_path in temp_paths:
            Path(temp_path).unlink()
        
        # Stage 3: HeatMap Analysis
        stage3_result = heatmap_filter.process_event(
            motion_mask=motion_mask,
            original_image=sample_frames[1],
            save_overlay=False
        )
        
        assert "anomaly_score" in stage3_result
        assert "heat_zone" in stage3_result
        
        # Visualize combined result
        img = visualizer.visualize_stage_3(
            sample_frames[1], motion_boxes, stage3_result,
            "test_camera", "stage1_to_3_test"
        )
        
        print(f"✓ Stage 1→3 visualization saved: {img}")
    
    def test_full_pipeline_stages_1_to_4(self, motion_detector, heatmap_filter,
                                         anomaly_detector, sample_frames, visualizer):
        """Test full pipeline: Stage 1 → 3 → 4."""
        camera_id = "test_camera"
        alert_id = "full_pipeline_test_1_to_4"
        
        # Save frames temporarily
        temp_dir = Path("data/test_outputs/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_paths = [str(temp_dir / f"frame_{i}.jpg") for i in range(2)]
        for i, frame in enumerate(sample_frames[:2]):
            cv2.imwrite(temp_paths[i], frame)
        
        # Stage 1: Motion Detection
        motion_result = motion_detector.detect_motion(temp_paths)
        motion_mask = motion_result.motion_mask
        motion_boxes = [region.to_bbox() for region in motion_result.motion_regions]
        
        # Cleanup
        for temp_path in temp_paths:
            Path(temp_path).unlink()
        
        assert motion_result.motion_detected, "Motion should be detected in Stage 1"
        
        # Create pipeline (Stages 3-4)
        pipeline = create_pipeline(
            heatmap_filter=heatmap_filter,
            anomaly_detector=anomaly_detector,
            yolo_detector=None
        )
        
        # Process through pipeline (Stages 3-4)
        result = pipeline.process_alert(
            camera_id=camera_id,
            alert_id=alert_id,
            images=sample_frames,
            motion_mask=motion_mask,
            motion_boxes=motion_boxes
        )
        
        # Verify results
        assert result["stage_3_result"] is not None, "Stage 3 should execute"
        assert result["stage_4_result"] is not None, "Stage 4 should execute"
        assert result["final_decision"] in ["ESCALATE", "DROP"], "Should have final decision"
        
        # Generate visualizations
        viz_paths = {}
        
        # Stage 1 visualization
        img_stage1 = sample_frames[1].copy()
        for (x1, y1, x2, y2) in motion_boxes:
            cv2.rectangle(img_stage1, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lines = [
            f"Camera: {camera_id} | Alert: {alert_id}",
            "STAGE 1: Motion Detection",
            f"Motion Detected: YES",
            f"Motion Regions: {len(motion_boxes)}",
            f"Confidence: {motion_result.confidence:.3f}"
        ]
        img_stage1 = visualizer.add_text_overlay(img_stage1, lines, bg_color=(0, 50, 0))
        viz_paths["stage1"] = visualizer.output_dir / f"{alert_id}_stage1.jpg"
        cv2.imwrite(str(viz_paths["stage1"]), img_stage1)
        
        # Stage 3 visualization
        viz_paths["stage3"] = visualizer.visualize_stage_3(
            sample_frames[1], motion_boxes, result["stage_3_result"],
            camera_id, alert_id
        )
        
        # Stage 4 visualization
        viz_paths["stage4"] = visualizer.visualize_stage_4(
            sample_frames[1], motion_boxes, result["stage_4_result"],
            camera_id, alert_id
        )
        
        # Final decision visualization
        viz_paths["final"] = visualizer.visualize_final_decision(
            sample_frames[1], result, motion_boxes, camera_id, alert_id
        )
        
        print("\n" + "=" * 70)
        print("Full Pipeline Test (Stages 1→3→4) Complete")
        print("=" * 70)
        print(f"Camera: {camera_id}")
        print(f"Alert ID: {alert_id}")
        print(f"\nStage 1 (Motion Detection):")
        print(f"  - Motion Detected: {motion_result.motion_detected}")
        print(f"  - Regions: {len(motion_boxes)}")
        print(f"  - Confidence: {motion_result.confidence:.3f}")
        print(f"\nStage 3 (HeatMap):")
        print(f"  - Heat Zone: {result['stage_3_result'].get('heat_zone')}")
        print(f"  - Anomaly Score: {result['stage_3_result'].get('anomaly_score'):.3f}")
        print(f"\nStage 4 (Memory-Based):")
        print(f"  - Decision: {result['stage_4_result'].get('decision')}")
        print(f"  - Similarity: {result['stage_4_result'].get('event_sim'):.3f}")
        print(f"  - Support: {result['stage_4_result'].get('event_support')}")
        print(f"\nFinal Decision: {result['final_decision']}")
        print(f"\nVisualizations saved:")
        for stage, path in viz_paths.items():
            print(f"  - {stage}: {path}")
        print("=" * 70 + "\n")
    
    def test_full_pipeline_with_yolo(self, motion_detector, heatmap_filter,
                                     anomaly_detector, yolo_detector,
                                     sample_frames, visualizer):
        """Test full pipeline: Stage 1 → 3 → 4 → 5 (with YOLO)."""
        camera_id = "test_camera"
        alert_id = "full_pipeline_test_with_yolo"
        
        # Save frames temporarily
        temp_dir = Path("data/test_outputs/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_paths = [str(temp_dir / f"frame_{i}.jpg") for i in range(2)]
        for i, frame in enumerate(sample_frames[:2]):
            cv2.imwrite(temp_paths[i], frame)
        
        # Stage 1: Motion Detection
        motion_result = motion_detector.detect_motion(temp_paths)
        motion_mask = motion_result.motion_mask
        motion_boxes = [region.to_bbox() for region in motion_result.motion_regions]
        
        # Cleanup
        for temp_path in temp_paths:
            Path(temp_path).unlink()
        
        # Create pipeline with YOLO (Stages 3-5)
        pipeline = create_pipeline(
            heatmap_filter=heatmap_filter,
            anomaly_detector=anomaly_detector,
            yolo_detector=yolo_detector
        )
        
        # Process through pipeline
        result = pipeline.process_alert(
            camera_id=camera_id,
            alert_id=alert_id,
            images=sample_frames,
            motion_mask=motion_mask,
            motion_boxes=motion_boxes
        )
        
        # Verify all stages executed
        assert result["stage_3_result"] is not None
        assert result["stage_4_result"] is not None
        # Stage 5 may or may not execute depending on Stage 4 decision
        
        # Generate all visualizations
        viz_paths = {}
        
        # Stage 1
        img_stage1 = sample_frames[1].copy()
        for (x1, y1, x2, y2) in motion_boxes:
            cv2.rectangle(img_stage1, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lines = [
            f"Camera: {camera_id} | Alert: {alert_id}",
            "STAGE 1: Motion Detection",
            f"Motion Detected: YES",
            f"Motion Regions: {len(motion_boxes)}"
        ]
        img_stage1 = visualizer.add_text_overlay(img_stage1, lines, bg_color=(0, 50, 0))
        viz_paths["stage1"] = visualizer.output_dir / f"{alert_id}_stage1.jpg"
        cv2.imwrite(str(viz_paths["stage1"]), img_stage1)
        
        # Stage 3
        viz_paths["stage3"] = visualizer.visualize_stage_3(
            sample_frames[1], motion_boxes, result["stage_3_result"],
            camera_id, alert_id
        )
        
        # Stage 4
        viz_paths["stage4"] = visualizer.visualize_stage_4(
            sample_frames[1], motion_boxes, result["stage_4_result"],
            camera_id, alert_id
        )
        
        # Stage 5 (if executed)
        if result["stage_5_result"]:
            viz_paths["stage5"] = visualizer.visualize_stage_5(
                sample_frames[1], result["stage_5_result"],
                camera_id, alert_id
            )
        
        # Final
        viz_paths["final"] = visualizer.visualize_final_decision(
            sample_frames[1], result, motion_boxes, camera_id, alert_id
        )
        
        print("\n" + "=" * 70)
        print("Full Pipeline Test (Stages 1→3→4→5) Complete")
        print("=" * 70)
        print(f"Final Decision: {result['final_decision']}")
        print(f"Stage 5 Executed: {result['stage_5_result'] is not None}")
        print(f"\nVisualizations saved:")
        for stage, path in viz_paths.items():
            print(f"  - {stage}: {path}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
