"""
Quick test with real training data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pytest
from core.motion_detection import MotionDetector

# Resolve repo root and data path (Factory camera events)
REPO_ROOT = Path(__file__).resolve().parents[2]
base_path = REPO_ROOT / "data" / "training" / "camera-events" / "Factory"

if not base_path.exists():
    pytest.skip(f"Missing data directory: {base_path}", allow_module_level=True)

# Find first available event
event_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])

if not event_dirs:
    print("No training data found!")
    exit(1)

# Use first event
event_dir = event_dirs[0]
images = sorted([str(p) for p in event_dir.glob("*.jpg")])

print(f"🔍 Testing Motion Detection")
print(f"Event: {event_dir.name}")
print(f"Images: {len(images)}")
print("=" * 60)

# Create detector
detector = MotionDetector(threshold=25, min_area=500)

# Detect motion
result = detector.detect_motion(images)

print(f"\n✓ Motion Detected: {result.motion_detected}")
print(f"  Confidence: {result.confidence:.2%}")
print(f"  Total Motion Area: {result.total_motion_area:,} pixels")
print(f"  Frame Size: {result.frame_dimensions[1]}x{result.frame_dimensions[0]}")
print(f"  Number of Regions: {len(result.motion_regions)}")

# Print regions
for i, region in enumerate(result.motion_regions[:5], 1):
    print(f"\n  Region #{i}:")
    print(f"    Bounding Box: ({region.x}, {region.y}) -> ({region.x + region.width}, {region.y + region.height})")
    print(f"    Size: {region.width}x{region.height}")
    print(f"    Area: {region.area:,} pixels")
    print(f"    Centroid: {region.centroid}")

# Create visualization
output_path = "../data/test_visualization.jpg"
vis = detector.visualize_motion(
    images[1],
    result,
    output_path=output_path,
    show_mask=True,
    show_boxes=True,
    show_centroids=True
)

print(f"\n✓ Visualization saved to: {output_path}")

# Save JSON in repo data folder
json_path = REPO_ROOT / "data" / "test_result.json"
with open(json_path, 'w') as f:
    json.dump(result.to_dict(), f, indent=2)

print(f"✓ JSON result saved to: {json_path}")
print("\n" + "=" * 60)
print("✓ Test completed successfully!")
