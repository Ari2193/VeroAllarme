"""
Run relevance demo on multiple events to show both relevant and irrelevant examples
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.motion_detection import MotionDetector
from core.relevance_mask import RelevanceMaskPredictor
import cv2
import numpy as np

print("=" * 80)
print("🎯 VeroAllarme - Multiple Events Demo")
print("=" * 80)

# Setup
data_dir = Path("../../data/training/camera-events/Factory")
output_dir = Path("../../data/relevance_outputs")
output_dir.mkdir(exist_ok=True)

# Initialize
motion_detector = MotionDetector(threshold=25, min_area=500)
predictor = RelevanceMaskPredictor(relevance_threshold=0.5)

# Get events
events = sorted([d for d in data_dir.iterdir() if d.is_dir()])[:5]

print(f"\n📁 Processing {len(events)} events...")

results_summary = []

for i, event_dir in enumerate(events, 1):
    images = sorted([str(p) for p in event_dir.glob("*.jpg")])
    
    if len(images) < 2:
        continue
    
    print(f"\n[{i}/{len(events)}] {event_dir.name}")
    
    # Motion detection
    motion_result = motion_detector.detect_motion(images)
    
    if not motion_result.motion_detected:
        print("  ⚪ No motion detected")
        continue
    
    # Relevance prediction
    middle_frame = images[len(images) // 2]
    relevance_result = predictor.predict(middle_frame, motion_result)
    
    # Visualize
    vis_path = output_dir / f"event_{i}_{event_dir.name}.jpg"
    frame = cv2.imread(middle_frame)
    predictor.visualize(frame, relevance_result, output_path=str(vis_path))
    
    # Summary
    if relevance_result.num_relevant > 0:
        status = "🟢 RELEVANT"
        max_score = max(r.relevance_score for r in relevance_result.relevant_regions)
    else:
        status = "🔴 IRRELEVANT"
        max_score = max(r.relevance_score for r in relevance_result.irrelevant_regions) if relevance_result.irrelevant_regions else 0
    
    print(f"  {status} - Relevant: {relevance_result.num_relevant}, "
          f"Irrelevant: {relevance_result.num_irrelevant}, "
          f"Max Score: {max_score:.2f}")
    
    results_summary.append({
        "event": event_dir.name,
        "status": status,
        "relevant": relevance_result.num_relevant,
        "irrelevant": relevance_result.num_irrelevant,
        "image": vis_path.name
    })

# Create summary visualization
print("\n" + "=" * 80)
print("📊 Summary")
print("=" * 80)

for result in results_summary:
    print(f"\n{result['status']} {result['event']}")
    print(f"  Relevant: {result['relevant']}, Irrelevant: {result['irrelevant']}")
    print(f"  Image: {result['image']}")

print(f"\n✓ All visualizations saved to: {output_dir.absolute()}")
print("\n💡 Open the images to see:")
print("  🟢 Green boxes = RELEVANT alerts (should trigger notification)")
print("  🔴 Red boxes = IRRELEVANT motion (false alarms to filter)")
print("=" * 80)
