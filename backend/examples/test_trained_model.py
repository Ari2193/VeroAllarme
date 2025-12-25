"""
Test the trained model on sample events
Compare with untrained model predictions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pytest
from core.motion_detection import MotionDetector
from core.relevance_mask import RelevanceMaskPredictor

print("=" * 80)
print("🧪 Testing Trained Model")
print("=" * 80)

REPO_ROOT = Path(__file__).resolve().parents[2]
data_dir = REPO_ROOT / "data" / "training" / "camera-events" / "Factory"
model_path = REPO_ROOT / "backend" / "model_checkpoints" / "quick_complete.pth"

if not data_dir.exists():
    pytest.skip(f"Missing data directory: {data_dir}", allow_module_level=True)

if not model_path.exists():
    pytest.skip(f"Missing trained model checkpoint: {model_path}", allow_module_level=True)

# Initialize detectors
motion_detector = MotionDetector()

# Test both models
untrained_predictor = RelevanceMaskPredictor()  # Random weights
trained_predictor = RelevanceMaskPredictor(
    model_path=str(model_path)
)

print("\n✓ Loaded models (trained vs untrained)")

# Test on multiple events
event_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])[:5]

print(f"\n📋 Testing on {len(event_dirs)} events...")

results = []

for i, event_dir in enumerate(event_dirs, 1):
    event_name = event_dir.name
    
    # Get images
    images = sorted(event_dir.glob("*.jpg"))
    if len(images) < 2:
        continue
    
    # Load frame
    frame = cv2.imread(str(images[len(images) // 2]))
    
    # Detect motion
    motion_result = motion_detector.detect_motion([str(p) for p in images[:5]])
    
    if not motion_result.motion_detected:
        continue
    
    print(f"\n{i}. Event: {event_name}")
    print(f"   Motion regions: {len(motion_result.motion_regions)}")
    
    # Get predictions
    untrained_result = untrained_predictor.predict(frame, motion_result)
    trained_result = trained_predictor.predict(frame, motion_result)
    
    untrained_relevant = sum(1 for r in untrained_result.regions if r.is_relevant)
    trained_relevant = sum(1 for r in trained_result.regions if r.is_relevant)
    
    untrained_avg_score = np.mean([r.relevance_score for r in untrained_result.regions])
    trained_avg_score = np.mean([r.relevance_score for r in trained_result.regions])
    
    print(f"   Untrained: {untrained_relevant}/{len(untrained_result.regions)} relevant, "
          f"avg score: {untrained_avg_score:.3f}")
    print(f"   Trained:   {trained_relevant}/{len(trained_result.regions)} relevant, "
          f"avg score: {trained_avg_score:.3f}")
    
    # Determine if predictions differ
    same_prediction = (untrained_relevant > 0) == (trained_relevant > 0)
    
    results.append({
        "event": event_name,
        "motion_regions": len(motion_result.motion_regions),
        "untrained_relevant": untrained_relevant,
        "trained_relevant": trained_relevant,
        "untrained_score": float(untrained_avg_score),
        "trained_score": float(trained_avg_score),
        "same_prediction": same_prediction
    })

# Summary
print("\n" + "=" * 80)
print("📊 SUMMARY")
print("=" * 80)

total_events = len(results)
different_predictions = sum(1 for r in results if not r["same_prediction"])

print(f"\nTotal events tested: {total_events}")
print(f"Different predictions: {different_predictions}/{total_events} "
      f"({different_predictions/total_events*100:.1f}%)")

# Count relevant predictions
untrained_total_relevant = sum(r["untrained_relevant"] for r in results)
trained_total_relevant = sum(r["trained_relevant"] for r in results)
total_regions = sum(r["motion_regions"] for r in results)

print(f"\nRelevant regions marked:")
print(f"  Untrained: {untrained_total_relevant}/{total_regions} "
      f"({untrained_total_relevant/total_regions*100:.1f}%)")
print(f"  Trained:   {trained_total_relevant}/{total_regions} "
      f"({trained_total_relevant/total_regions*100:.1f}%)")

# Average scores
avg_untrained = np.mean([r["untrained_score"] for r in results])
avg_trained = np.mean([r["trained_score"] for r in results])

print(f"\nAverage relevance scores:")
print(f"  Untrained: {avg_untrained:.3f}")
print(f"  Trained:   {avg_trained:.3f}")

print("\n" + "=" * 80)
print("✅ Analysis complete!")
print("=" * 80)

# Interpretation
print("\n💡 INTERPRETATION:")
if abs(avg_trained - 0.5) > abs(avg_untrained - 0.5):
    print("✓ Trained model is more confident in its predictions")
else:
    print("⚠ Model needs more training for confident predictions")

if different_predictions > 0:
    print(f"✓ Training affected {different_predictions} event classifications")
else:
    print("⚠ Training didn't change any classifications significantly")

print("\nNote: Model was trained on only 50 sequences for 5 epochs.")
print("For better results, train on full dataset with 20+ epochs.")
