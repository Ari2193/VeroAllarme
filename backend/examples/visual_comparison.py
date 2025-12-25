"""
Visual comparison: Trained vs Untrained model predictions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from core.motion_detection import MotionDetector
from core.relevance_mask import RelevanceMaskPredictor

print("=" * 80)
print("🎨 Creating Visual Comparison")
print("=" * 80)

# Initialize
motion_detector = MotionDetector()
untrained = RelevanceMaskPredictor()
trained = RelevanceMaskPredictor(model_path="model_checkpoints/quick_complete.pth")

# Test event
data_dir = Path("../data/training/camera-events/Factory")
event_dir = next(d for d in data_dir.iterdir() if d.is_dir())
images = sorted(event_dir.glob("*.jpg"))

print(f"\nProcessing: {event_dir.name}")

# Load frame
frame = cv2.imread(str(images[len(images) // 2]))
h, w = frame.shape[:2]

# Detect motion
motion_result = motion_detector.detect_motion([str(p) for p in images[:5]])

if not motion_result.motion_detected:
    print("No motion detected!")
    exit(1)

print(f"Motion regions: {len(motion_result.motion_regions)}")

# Get predictions
untrained_result = untrained.predict(frame, motion_result)
trained_result = trained.predict(frame, motion_result)

# Create visualization
vis_frame = frame.copy()
canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)

# Column 1: Motion detection
motion_vis = frame.copy()
for region in motion_result.motion_regions:
    x1, y1, x2, y2 = region.to_bbox()
    cv2.rectangle(motion_vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.putText(motion_vis, "MOTION", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

# Column 2: Untrained predictions
untrained_vis = frame.copy()
for r in untrained_result.regions:
    x1, y1, x2, y2 = r.to_bbox()
    color = (0, 255, 0) if r.is_relevant else (0, 0, 255)
    label = "REL" if r.is_relevant else "IRREL"
    cv2.rectangle(untrained_vis, (x1, y1), (x2, y2), color, 2)
    cv2.putText(untrained_vis, f"{label} {r.relevance_score:.2f}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Column 3: Trained predictions
trained_vis = frame.copy()
for r in trained_result.regions:
    x1, y1, x2, y2 = r.to_bbox()
    color = (0, 255, 0) if r.is_relevant else (0, 0, 255)
    label = "REL" if r.is_relevant else "IRREL"
    cv2.rectangle(trained_vis, (x1, y1), (x2, y2), color, 2)
    cv2.putText(trained_vis, f"{label} {r.relevance_score:.2f}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Combine
canvas[:, :w] = motion_vis
canvas[:, w:w*2] = untrained_vis
canvas[:, w*2:] = trained_vis

# Add titles
title_h = 60
titled_canvas = np.zeros((h + title_h, w * 3, 3), dtype=np.uint8)
titled_canvas[title_h:] = canvas

# Add title text
titles = [
    ("Stage 1: Motion Detection", w//2),
    ("Untrained Model", w + w//2),
    ("Trained Model (5 epochs)", 2*w + w//2)
]

for title, x in titles:
    # Get text size for centering
    (text_w, text_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_x = x - text_w // 2
    cv2.putText(titled_canvas, title, (text_x, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Add dividers
cv2.line(titled_canvas, (w, title_h), (w, h + title_h), (255, 255, 255), 2)
cv2.line(titled_canvas, (w*2, title_h), (w*2, h + title_h), (255, 255, 255), 2)

# Save
output_path = Path("data/relevance_outputs/trained_comparison.jpg")
output_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(output_path), titled_canvas)

print(f"\n✅ Saved comparison: {output_path}")

# Print stats
print("\n" + "=" * 80)
print("📊 STATISTICS")
print("=" * 80)

untrained_relevant = sum(1 for r in untrained_result.regions if r.is_relevant)
trained_relevant = sum(1 for r in trained_result.regions if r.is_relevant)

untrained_avg = np.mean([r.relevance_score for r in untrained_result.regions])
trained_avg = np.mean([r.relevance_score for r in trained_result.regions])

print(f"\nMotion Detection:")
print(f"  Regions: {len(motion_result.motion_regions)}")
print(f"  Total area: {motion_result.total_motion_area} pixels")

print(f"\nUntrained Model:")
print(f"  Relevant: {untrained_relevant}/{len(untrained_result.regions)}")
print(f"  Average score: {untrained_avg:.3f}")

print(f"\nTrained Model:")
print(f"  Relevant: {trained_relevant}/{len(trained_result.regions)}")
print(f"  Average score: {trained_avg:.3f}")

score_diff = trained_avg - untrained_avg
print(f"\nScore improvement: {score_diff:+.3f} ({score_diff/untrained_avg*100:+.1f}%)")

print("\n" + "=" * 80)
print(f"View the comparison image: {output_path}")
print("=" * 80)
