"""
Test location relevance model on random images
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import random
from core.location_relevance import LocationRelevancePredictor

print("=" * 80)
print("🎯 Testing Location Relevance on Random Images")
print("=" * 80)

# Load predictor
predictor = LocationRelevancePredictor(camera_name="Factory")
print("✓ Model loaded\n")

# Find all images
data_dir = Path("../data/training/camera-events/Factory")
all_images = []

for event_dir in data_dir.iterdir():
    if event_dir.is_dir():
        images = list(event_dir.glob("*.jpg"))
        all_images.extend(images)

print(f"✓ Found {len(all_images)} total images")

# Pick 5 random
random.seed(42)
selected_images = random.sample(all_images, min(5, len(all_images)))

print(f"✓ Selected {len(selected_images)} random images\n")
print("=" * 80)

# Process each
output_dir = Path("data/relevance_outputs/test_samples")
output_dir.mkdir(parents=True, exist_ok=True)

for i, img_path in enumerate(selected_images, 1):
    print(f"\n📷 Image {i}: {img_path.parent.name}/{img_path.name}")
    
    # Load frame
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"   ❌ Failed to load")
        continue
    
    h, w = frame.shape[:2]
    print(f"   Size: {w}x{h}")
    
    # Generate heat map overlay
    visualization = predictor.visualize_heat_map(frame, alpha=0.5)
    
    # Save
    output_path = output_dir / f"test_{i}_{img_path.parent.name}_{img_path.name}"
    cv2.imwrite(str(output_path), visualization)
    print(f"   ✅ Saved: {output_path}")
    
    # Test some locations
    test_points = [
        ("Top", 0.5, 0.1),
        ("Middle", 0.5, 0.5),
        ("Bottom", 0.5, 0.9)
    ]
    
    print("   Location tests:")
    for name, nx, ny in test_points:
        is_rel, score = predictor.is_location_relevant(
            int(nx * w), int(ny * h), w, h
        )
        status = "✓ RELEVANT" if is_rel else "✗ NOT RELEVANT"
        print(f"      {name:8s}: {status:15s} (score: {score:.3f})")

print("\n" + "=" * 80)
print(f"✅ Complete! Saved {len(selected_images)} visualizations to:")
print(f"   {output_dir}")
print("=" * 80)
