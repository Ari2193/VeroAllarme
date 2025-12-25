"""
Train and demo location-based relevance model
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.location_relevance import LocationRelevanceTrainer, LocationRelevancePredictor
import cv2
import numpy as np

print("=" * 80)
print("🎯 Location-Based Relevance Training & Demo")
print("=" * 80)

# ==================== TRAINING ====================

print("\n📚 STEP 1: Training Location Model")
print("-" * 80)

trainer = LocationRelevanceTrainer(
    camera_name="Factory",
    data_dir="../data/training/camera-events/Factory"
)

# Train on motion data to learn relevant locations
trainer.train(
    epochs=15,
    batch_size=256,
    samples_per_event=200  # 200 random samples per event
)

# ==================== DEMO ====================

print("\n\n🎨 STEP 2: Generating Heat Map Visualization")
print("-" * 80)

# Load trained model
predictor = LocationRelevancePredictor(camera_name="Factory")

# Load a sample frame
data_dir = Path("../data/training/camera-events/Factory")
event_dir = next(d for d in data_dir.iterdir() if d.is_dir())
images = sorted(event_dir.glob("*.jpg"))
frame = cv2.imread(str(images[0]))

print(f"\nProcessing frame from: {event_dir.name}")
print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")

# Generate heat map
heat_map = predictor.get_heat_map(
    frame_width=frame.shape[1],
    frame_height=frame.shape[0],
    resolution=(224, 224)
)

print(f"✓ Generated heat map: {heat_map.shape}")
print(f"   Min score: {heat_map.min():.3f}")
print(f"   Max score: {heat_map.max():.3f}")
print(f"   Mean score: {heat_map.mean():.3f}")

# Visualize
visualization = predictor.visualize_heat_map(frame, alpha=0.6)

# Save
output_dir = Path("data/relevance_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "location_heat_map.jpg"
cv2.imwrite(str(output_path), visualization)

print(f"\n✅ Saved heat map visualization: {output_path}")

# ==================== LOCATION TESTS ====================

print("\n\n🧪 STEP 3: Testing Specific Locations")
print("-" * 80)

h, w = frame.shape[:2]

# Test various locations
test_locations = [
    ("Top-left corner", 100, 100),
    ("Top-right corner", w - 100, 100),
    ("Center", w // 2, h // 2),
    ("Bottom-left", 100, h - 100),
    ("Bottom-right", w - 100, h - 100),
]

print("\nLocation Relevance Tests:")
for name, x, y in test_locations:
    is_relevant, score = predictor.is_location_relevant(x, y, w, h)
    status = "✓ RELEVANT" if is_relevant else "✗ NOT RELEVANT"
    print(f"  {name:20s} ({x:4d}, {y:4d}): {status:15s} (score: {score:.3f})")

# ==================== STATISTICS ====================

print("\n\n📊 STEP 4: Overall Statistics")
print("-" * 80)

relevant_pixels = (heat_map > 0.5).sum()
total_pixels = heat_map.size
relevant_percentage = (relevant_pixels / total_pixels) * 100

print(f"\nRelevant Coverage:")
print(f"  Relevant pixels: {relevant_pixels}/{total_pixels}")
print(f"  Coverage: {relevant_percentage:.1f}%")

# Create binary mask visualization
binary_mask = (heat_map > 0.5).astype(np.uint8) * 255
binary_mask_resized = cv2.resize(binary_mask, (w, h))

# Create side-by-side comparison
comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
comparison[:, :w] = frame
comparison[:, w:, 0] = binary_mask_resized  # Red channel for irrelevant
comparison[:, w:, 1] = 255 - binary_mask_resized  # Green channel for relevant

# Add titles
titled = np.zeros((h + 60, w * 2, 3), dtype=np.uint8)
titled[60:] = comparison

cv2.putText(titled, "Original Frame", (w // 2 - 100, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(titled, "Relevance Mask (Green=Relevant)", (w + w // 2 - 200, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

comparison_path = output_dir / "location_mask_comparison.jpg"
cv2.imwrite(str(comparison_path), titled)

print(f"\n✅ Saved binary mask comparison: {comparison_path}")

print("\n" + "=" * 80)
print("✅ DEMO COMPLETE!")
print("=" * 80)
print(f"\nGenerated files:")
print(f"  1. {output_path}")
print(f"  2. {comparison_path}")
print(f"\nUsage in pipeline:")
print(f"  predictor = LocationRelevancePredictor(camera_name='Factory')")
print(f"  is_relevant, score = predictor.is_location_relevant(x, y, width, height)")
print("=" * 80)
