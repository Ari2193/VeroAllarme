"""
Batch test on 10 different events from training data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.motion_detection import MotionDetector
import json
from datetime import datetime

print("=" * 80)
print("🔍 Motion Detection Batch Test - 10 Events")
print("=" * 80)

# Find events from all cameras
cameras = ["Factory", "Field", "Gate and rode", "Trees"]
all_events = []

for camera in cameras:
    base_path = Path(f"../../data/training/camera-events/{camera}")
    if not base_path.exists():
        continue
    
    events = sorted([d for d in base_path.iterdir() if d.is_dir()])
    for event in events[:3]:  # Take 3 events from each camera
        images = sorted([str(p) for p in event.glob("*.jpg")])
        if len(images) >= 2:
            all_events.append({
                "camera": camera,
                "event": event.name,
                "images": images
            })

# Limit to 10 events
all_events = all_events[:10]

print(f"\nTesting {len(all_events)} events from {len(set(e['camera'] for e in all_events))} cameras")
print("=" * 80)

# Create detector
detector = MotionDetector(threshold=25, min_area=500)

results_summary = []

for i, event_data in enumerate(all_events, 1):
    camera = event_data["camera"]
    event = event_data["event"]
    images = event_data["images"]
    
    print(f"\n[{i}/10] Testing: {camera}/{event}")
    print(f"  Images: {len(images)}")
    
    try:
        # Detect motion
        start_time = datetime.now()
        result = detector.detect_motion(images)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Print results
        status = "✓ MOTION" if result.motion_detected else "✗ No Motion"
        print(f"  {status}")
        print(f"  Processing Time: {processing_time:.1f}ms")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Motion Area: {result.total_motion_area:,} pixels")
        print(f"  Regions: {len(result.motion_regions)}")
        
        # Show top 3 regions
        if result.motion_regions:
            for j, region in enumerate(result.motion_regions[:3], 1):
                print(f"    Region {j}: ({region.x}, {region.y}) "
                      f"{region.width}x{region.height} = {region.area:,}px")
        
        # Create visualization for first 3 events
        if i <= 3:
            vis_path = f"../../data/visualization_{i}_{camera}_{event}.jpg"
            detector.visualize_motion(
                images[1],
                result,
                output_path=vis_path,
                show_mask=True,
                show_boxes=True,
                show_centroids=True
            )
            print(f"  📸 Saved: {vis_path}")
        
        # Store summary
        results_summary.append({
            "index": i,
            "camera": camera,
            "event": event,
            "motion_detected": result.motion_detected,
            "confidence": float(result.confidence),
            "total_area": result.total_motion_area,
            "num_regions": len(result.motion_regions),
            "processing_time_ms": round(processing_time, 1),
            "frame_size": f"{result.frame_dimensions[1]}x{result.frame_dimensions[0]}"
        })
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        results_summary.append({
            "index": i,
            "camera": camera,
            "event": event,
            "error": str(e)
        })

# Print summary
print("\n" + "=" * 80)
print("📊 SUMMARY")
print("=" * 80)

successful = [r for r in results_summary if "error" not in r]
motion_detected = [r for r in successful if r["motion_detected"]]

print(f"\nTotal Events Processed: {len(successful)}/{len(all_events)}")
if len(successful) > 0:
    print(f"Motion Detected: {len(motion_detected)}/{len(successful)} ({len(motion_detected)/len(successful)*100:.1f}%)")
else:
    print("No events found or processed")

if successful:
    avg_confidence = sum(r["confidence"] for r in successful) / len(successful)
    avg_time = sum(r["processing_time_ms"] for r in successful) / len(successful)
    total_regions = sum(r["num_regions"] for r in motion_detected)
    
    print(f"Average Confidence: {avg_confidence:.2%}")
    print(f"Average Processing Time: {avg_time:.1f}ms")
    print(f"Total Regions Found: {total_regions}")

# Print detailed table
print("\n" + "=" * 80)
print("DETAILED RESULTS")
print("=" * 80)
print(f"{'#':<3} {'Camera':<18} {'Event':<18} {'Motion':<8} {'Regions':<8} {'Time(ms)':<10}")
print("-" * 80)

for r in results_summary:
    if "error" in r:
        print(f"{r['index']:<3} {r['camera']:<18} {r['event']:<18} ERROR")
    else:
        motion_str = "✓ Yes" if r['motion_detected'] else "✗ No"
        print(f"{r['index']:<3} {r['camera']:<18} {r['event']:<18} "
              f"{motion_str:<8} {r['num_regions']:<8} {r['processing_time_ms']:<10.1f}")

# Save JSON summary
json_path = "../../data/batch_test_results.json"
with open(json_path, 'w') as f:
    json.dump({
        "test_date": datetime.now().isoformat(),
        "total_events": len(all_events),
        "successful": len(successful),
        "motion_detected": len(motion_detected),
        "results": results_summary
    }, f, indent=2)

print(f"\n✓ Results saved to: {json_path}")
print("=" * 80)
