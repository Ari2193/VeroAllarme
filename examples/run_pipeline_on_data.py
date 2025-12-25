"""
Run the full pipeline on real camera event data and generate visualizations.

This script:
1. Loads events from data/training/camera-events/
2. Processes them through the complete pipeline (Stages 3, 4, 5)
3. Generates annotated images for each stage
4. Saves results to a report file
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2

from backend.core.pipeline import create_pipeline
from backend.core.anomaly_detection import AnomalyDetector
from backend.core.pipeline_visualizer import PipelineVisualizer

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


def generate_simple_motion_boxes(image: np.ndarray, num_boxes: int = 2):
    """Generate simple motion boxes for testing (since we don't have motion masks)."""
    h, w = image.shape[:2]
    boxes = []
    
    # Generate boxes in different regions
    for i in range(num_boxes):
        x1 = np.random.randint(0, w // 2)
        y1 = np.random.randint(0, h // 2)
        x2 = x1 + np.random.randint(50, 150)
        y2 = y1 + np.random.randint(50, 150)
        
        # Ensure boxes are within image bounds
        x2 = min(x2, w - 1)
        y2 = min(y2, h - 1)
        
        boxes.append((x1, y1, x2, y2))
    
    return boxes


def create_motion_mask_from_boxes(boxes, shape):
    """Create a binary motion mask from bounding boxes."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for (x1, y1, x2, y2) in boxes:
        mask[y1:y2, x1:x2] = 255
    return mask


def process_event(pipeline, visualizer, event_path: Path, camera_id: str):
    """Process a single event through the pipeline."""
    # Load images
    images = load_event_images(event_path)
    if images is None:
        print(f"  ⚠️  Could not load 3 images from {event_path.name}")
        return None
    
    print(f"  ✓ Loaded 3 images from {event_path.name}")
    
    # Generate motion boxes and mask (simulated)
    motion_boxes = generate_simple_motion_boxes(images[1], num_boxes=2)
    motion_mask = create_motion_mask_from_boxes(motion_boxes, images[1].shape)
    
    alert_id = f"{camera_id}_{event_path.name}"
    
    # Process through pipeline
    print(f"  → Processing through pipeline...")
    result = pipeline.process_alert(
        camera_id=camera_id,
        alert_id=alert_id,
        images=images,
        motion_mask=motion_mask,
        motion_boxes=motion_boxes,
    )
    
    # Generate visualizations
    print(f"  → Generating visualizations...")
    
    viz_paths = {}
    
    # Stage 3 visualization
    if result["stage_3_result"]:
        viz_paths["stage3"] = visualizer.visualize_stage_3(
            images[1], motion_boxes, result["stage_3_result"],
            camera_id, alert_id
        )
    
    # Stage 4 visualization
    if result["stage_4_result"]:
        viz_paths["stage4"] = visualizer.visualize_stage_4(
            images[1], motion_boxes, result["stage_4_result"],
            camera_id, alert_id
        )
    
    # Stage 5 visualization (if ran)
    if result["stage_5_result"]:
        viz_paths["stage5"] = visualizer.visualize_stage_5(
            images[1], result["stage_5_result"],
            camera_id, alert_id
        )
    
    # Final decision visualization
    viz_paths["final"] = visualizer.visualize_final_decision(
        images[1], result, motion_boxes, camera_id, alert_id
    )
    
    result["visualization_paths"] = {k: str(v) for k, v in viz_paths.items()}
    
    print(f"  ✓ Final Decision: {result['final_decision']}")
    print(f"  ✓ Visualizations saved")
    
    return result


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("Pipeline Execution on Real Camera Event Data")
    print("=" * 80 + "\n")
    
    # Setup paths
    data_root = Path("data/training/camera-events")
    output_dir = Path("data/pipeline_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline components
    print("Initializing pipeline components...")
    heatmap_filter = FilterHeatMap(camera_id="default", frame_shape=(480, 640))
    anomaly_detector = AnomalyDetector()
    pipeline = create_pipeline(
        heatmap_filter=heatmap_filter,
        anomaly_detector=anomaly_detector,
        yolo_detector=None  # YOLO optional
    )
    
    visualizer = PipelineVisualizer(output_dir=str(output_dir))
    print("✓ Pipeline ready\n")
    
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
        
        # Process first 3 events from this location
        for event_dir in event_dirs[:3]:
            print(f"\n  Event: {event_dir.name}")
            result = process_event(pipeline, visualizer, event_dir, location)
            
            if result:
                all_results.append({
                    "location": location,
                    "event": event_dir.name,
                    "alert_id": result["alert_id"],
                    "final_decision": result["final_decision"],
                    "explanation": result["explanation"],
                    "stage_3_heat_zone": result["stage_3_result"].get("heat_zone", "N/A"),
                    "stage_3_anomaly_score": result["stage_3_result"].get("anomaly_score", 0.0),
                    "stage_4_decision": result["stage_4_result"].get("decision", "N/A"),
                    "stage_4_similarity": result["stage_4_result"].get("event_sim", 0.0),
                    "stage_4_support": result["stage_4_result"].get("event_support", 0),
                    "visualization_paths": result.get("visualization_paths", {}),
                })
    
    # Save results to JSON
    results_file = output_dir / "pipeline_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_events_processed": len(all_results),
            "results": all_results
        }, f, indent=2)
    
    print(f"\n\n{'=' * 80}")
    print(f"✓ Processing complete!")
    print(f"{'=' * 80}")
    print(f"Total events processed: {len(all_results)}")
    print(f"Results saved to: {results_file}")
    print(f"Visualizations saved to: {output_dir}")
    
    # Summary statistics
    escalated = sum(1 for r in all_results if r["final_decision"] == "ESCALATE")
    dropped = sum(1 for r in all_results if r["final_decision"] == "DROP")
    
    print(f"\nDecision Summary:")
    print(f"  ESCALATE: {escalated} ({escalated/len(all_results)*100:.1f}%)")
    print(f"  DROP:     {dropped} ({dropped/len(all_results)*100:.1f}%)")
    
    # Generate text report
    report_file = output_dir / "pipeline_report.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Pipeline Execution Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Events Processed: {len(all_results)}\n\n")
        
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
            f.write(f"   Final Decision: {result['final_decision']}\n")
            f.write(f"\n   Stage 3 (HeatMap):\n")
            f.write(f"     - Heat Zone: {result['stage_3_heat_zone']}\n")
            f.write(f"     - Anomaly Score: {result['stage_3_anomaly_score']:.3f}\n")
            f.write(f"\n   Stage 4 (Memory-Based):\n")
            f.write(f"     - Decision: {result['stage_4_decision']}\n")
            f.write(f"     - Similarity: {result['stage_4_similarity']:.3f}\n")
            f.write(f"     - Support: {result['stage_4_support']}\n")
            f.write(f"\n   Explanation:\n")
            f.write(f"     {result['explanation']}\n")
            f.write(f"\n   Visualizations:\n")
            for stage, path in result['visualization_paths'].items():
                f.write(f"     - {stage}: {path}\n")
            f.write(f"\n   {'-' * 78}\n")
    
    print(f"Text report saved to: {report_file}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
