"""
Run Full Pipeline (Stages 1-5) on Real Camera Event Data

This script processes real camera events through all 5 stages:
1. Motion Detection
2. Masked Region Filtering (optional)
3. HeatMap Analysis
4. Memory-based Anomaly Detection
5. YOLO Object Detection

Generates annotated images for each stage showing the complete flow.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2

from backend.core.motion_detection import MotionDetector
from backend.core.pipeline import create_pipeline
from backend.core.anomaly_detection import AnomalyDetector
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


def load_event_images(event_path: Path):
    """Load the three images from an event directory."""
    images = []
    for i in range(3):
        img_path = event_path / f"{i}.jpg"
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
    return images if len(images) == 3 else None


def process_event_full_pipeline(motion_detector, pipeline, visualizer, 
                                event_path: Path, camera_id: str, use_yolo: bool = False):
    """Process a single event through all 5 stages."""
    # Load images
    images = load_event_images(event_path)
    if images is None:
        print(f"  ⚠️  Could not load 3 images from {event_path.name}")
        return None
    
    print(f"  ✓ Loaded 3 images from {event_path.name}")
    
    alert_id = f"{camera_id}_{event_path.name}"
    
    # ========== STAGE 1: Motion Detection ==========
    print(f"  → Stage 1: Motion Detection...")
    
    # Save images temporarily for motion detection
    temp_dir = Path("data/temp_motion")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_paths = []
    for i, img in enumerate(images[:2]):  # Use first 2 images
        temp_path = temp_dir / f"{alert_id}_{i}.jpg"
        cv2.imwrite(str(temp_path), img)
        temp_paths.append(str(temp_path))
    
    motion_result = motion_detector.detect_motion(temp_paths)
    
    # Cleanup temp files
    for temp_path in temp_paths:
        Path(temp_path).unlink()
    
    if not motion_result.motion_detected:
        print(f"  ⚠️  No motion detected, skipping event")
        return None
    
    motion_mask = motion_result.motion_mask
    motion_boxes = [region.to_bbox() for region in motion_result.motion_regions]
    
    print(f"     Motion detected: {len(motion_boxes)} regions")
    
    # ========== STAGES 3-5: Pipeline Processing ==========
    print(f"  → Stages 3-5: Pipeline processing...")
    pipeline_result = pipeline.process_alert(
        camera_id=camera_id,
        alert_id=alert_id,
        images=images,
        motion_mask=motion_mask,
        motion_boxes=motion_boxes,
    )
    
    # ========== VISUALIZATION ==========
    print(f"  → Generating visualizations...")
    
    viz_paths = {}
    
    # Stage 1 Visualization
    img_stage1 = images[1].copy()
    for (x1, y1, x2, y2) in motion_boxes:
        cv2.rectangle(img_stage1, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    lines = [
        f"Camera: {camera_id} | Alert: {alert_id}",
        f"STAGE 1: Motion Detection",
        f"Motion Detected: YES",
        f"Motion Regions: {len(motion_boxes)}",
        f"Total Motion Area: {motion_result.total_motion_area} pixels",
        f"Confidence: {motion_result.confidence:.3f}"
    ]
    img_stage1 = visualizer.add_text_overlay(img_stage1, lines, bg_color=(0, 50, 0))
    viz_paths["stage1"] = visualizer.output_dir / f"{alert_id}_stage1.jpg"
    cv2.imwrite(str(viz_paths["stage1"]), img_stage1)
    
    # Stage 3 Visualization
    if pipeline_result["stage_3_result"]:
        viz_paths["stage3"] = visualizer.visualize_stage_3(
            images[1], motion_boxes, pipeline_result["stage_3_result"],
            camera_id, alert_id
        )
    
    # Stage 4 Visualization
    if pipeline_result["stage_4_result"]:
        viz_paths["stage4"] = visualizer.visualize_stage_4(
            images[1], motion_boxes, pipeline_result["stage_4_result"],
            camera_id, alert_id
        )
    
    # Stage 5 Visualization (if ran)
    if pipeline_result["stage_5_result"]:
        viz_paths["stage5"] = visualizer.visualize_stage_5(
            images[1], pipeline_result["stage_5_result"],
            camera_id, alert_id
        )
    
    # Final Decision Visualization
    viz_paths["final"] = visualizer.visualize_final_decision(
        images[1], pipeline_result, motion_boxes, camera_id, alert_id
    )
    
    print(f"  ✓ Final Decision: {pipeline_result['final_decision']}")
    print(f"  ✓ Visualizations saved ({len(viz_paths)} images)")
    
    # Compile results
    result = {
        "location": camera_id,
        "event": event_path.name,
        "alert_id": alert_id,
        "stage_1_motion_detected": motion_result.motion_detected,
        "stage_1_motion_regions": len(motion_boxes),
        "stage_1_confidence": motion_result.confidence,
        "stage_3_heat_zone": pipeline_result["stage_3_result"].get("heat_zone", "N/A"),
        "stage_3_anomaly_score": pipeline_result["stage_3_result"].get("anomaly_score", 0.0),
        "stage_4_decision": pipeline_result["stage_4_result"].get("decision", "N/A"),
        "stage_4_similarity": pipeline_result["stage_4_result"].get("event_sim", 0.0),
        "stage_4_support": pipeline_result["stage_4_result"].get("event_support", 0),
        "stage_5_executed": pipeline_result["stage_5_result"] is not None,
        "final_decision": pipeline_result["final_decision"],
        "explanation": pipeline_result["explanation"],
        "visualization_paths": {k: str(v) for k, v in viz_paths.items()},
    }
    
    return result


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("Full Pipeline Execution (Stages 1-5) on Real Camera Event Data")
    print("=" * 80 + "\n")
    
    # Setup paths
    data_root = Path("data/training/camera-events")
    output_dir = Path("data/full_pipeline_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    print("Initializing pipeline components...")
    
    # Stage 1: Motion Detector
    motion_detector = MotionDetector(threshold=25, min_area=500)
    
    # Stage 3: HeatMap Filter
    heatmap_filter = FilterHeatMap(camera_id="default", frame_shape=(480, 640))
    
    # Stage 4: Anomaly Detector
    anomaly_detector = AnomalyDetector()
    
    # Stage 5: YOLO (optional)
    if YOLO_AVAILABLE:
        try:
            yolo_detector = YoloProcessor(model_name="yolov8n.pt")
            print("  ✓ YOLO detector loaded (Stage 5 enabled)")
            use_yolo = True
        except Exception as e:
            print(f"  ⚠️  YOLO not available: {e}")
            print("  → Stage 5 will be skipped")
            yolo_detector = None
            use_yolo = False
    else:
        print("  ⚠️  YOLO module not installed")
        print("  → Stage 5 will be skipped")
        yolo_detector = None
        use_yolo = False
    
    # Create pipeline (Stages 3-5)
    pipeline = create_pipeline(
        heatmap_filter=heatmap_filter,
        anomaly_detector=anomaly_detector,
        yolo_detector=yolo_detector
    )
    
    visualizer = PipelineVisualizer(output_dir=str(output_dir))
    print("✓ All components ready\n")
    
    # Collect results
    all_results = []
    
    # Process events from each camera location
    camera_locations = ["Factory", "Field", "Gate and rode", "Trees"]
    
    for location in camera_locations:
        location_path = data_root / location
        if not location_path.exists():
            continue
        
        print(f"\n📹 Processing location: {location}")
        print("-" * 60)
        
        # Get all event directories
        event_dirs = sorted([d for d in location_path.iterdir() if d.is_dir()])
        
        # Process first 2 events from this location
        for event_dir in event_dirs[:2]:
            print(f"\n  Event: {event_dir.name}")
            result = process_event_full_pipeline(
                motion_detector, pipeline, visualizer, 
                event_dir, location, use_yolo
            )
            
            if result:
                all_results.append(result)
    
    # Save results
    results_file = output_dir / "full_pipeline_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_events_processed": len(all_results),
            "yolo_enabled": use_yolo,
            "results": all_results
        }, f, indent=2)
    
    # Generate text report
    report_file = output_dir / "full_pipeline_report.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Full Pipeline Execution Report (Stages 1-5)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Events Processed: {len(all_results)}\n")
        f.write(f"YOLO Enabled (Stage 5): {use_yolo}\n\n")
        
        # Statistics
        escalated = sum(1 for r in all_results if r["final_decision"] == "ESCALATE")
        dropped = sum(1 for r in all_results if r["final_decision"] == "DROP")
        
        f.write(f"Decision Summary:\n")
        f.write(f"  ESCALATE: {escalated} ({escalated/len(all_results)*100:.1f}%)\n")
        f.write(f"  DROP:     {dropped} ({dropped/len(all_results)*100:.1f}%)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Detailed Results\n")
        f.write("=" * 80 + "\n\n")
        
        for i, result in enumerate(all_results, 1):
            f.write(f"\n{i}. Event: {result['event']}\n")
            f.write(f"   Location: {result['location']}\n")
            f.write(f"   Alert ID: {result['alert_id']}\n")
            f.write(f"\n   Stage 1 (Motion Detection):\n")
            f.write(f"     - Motion Detected: YES\n")
            f.write(f"     - Motion Regions: {result['stage_1_motion_regions']}\n")
            f.write(f"     - Confidence: {result['stage_1_confidence']:.3f}\n")
            f.write(f"\n   Stage 3 (HeatMap):\n")
            f.write(f"     - Heat Zone: {result['stage_3_heat_zone']}\n")
            f.write(f"     - Anomaly Score: {result['stage_3_anomaly_score']:.3f}\n")
            f.write(f"\n   Stage 4 (Memory-Based):\n")
            f.write(f"     - Decision: {result['stage_4_decision']}\n")
            f.write(f"     - Similarity: {result['stage_4_similarity']:.3f}\n")
            f.write(f"     - Support: {result['stage_4_support']}\n")
            f.write(f"\n   Stage 5 (YOLO):\n")
            f.write(f"     - Executed: {result['stage_5_executed']}\n")
            f.write(f"\n   Final Decision: {result['final_decision']}\n")
            f.write(f"\n   Explanation:\n")
            f.write(f"     {result['explanation']}\n")
            f.write(f"\n   Visualizations:\n")
            for stage, path in result['visualization_paths'].items():
                f.write(f"     - {stage}: {path}\n")
            f.write(f"\n   {'-' * 78}\n")
    
    # Print summary
    print(f"\n\n{'=' * 80}")
    print(f"✓ Processing complete!")
    print(f"{'=' * 80}")
    print(f"Total events processed: {len(all_results)}")
    print(f"YOLO enabled: {use_yolo}")
    print(f"Results saved to: {results_file}")
    print(f"Report saved to: {report_file}")
    print(f"Visualizations saved to: {output_dir}")
    
    # Summary statistics
    print(f"\nDecision Summary:")
    print(f"  ESCALATE: {escalated} ({escalated/len(all_results)*100:.1f}%)")
    print(f"  DROP:     {dropped} ({dropped/len(all_results)*100:.1f}%)")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
