"""
VeroAllarme - Quick Example: Relevance Mask Prediction

This script demonstrates how to use the full VeroAllarme pipeline:
Stage 1 (Motion Detection) + Stage 2 (Relevance Masking)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.motion_detection import MotionDetector
from core.relevance_mask import RelevanceMaskPredictor
import cv2
import json
from datetime import datetime


def main():
    print("=" * 80)
    print("🎯 VeroAllarme - Full Pipeline Demo")
    print("=" * 80)
    
    # Configuration
    data_dir = Path("../../data/training/camera-events/Factory")
    output_dir = Path("../../data/relevance_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Note: Model path should point to trained checkpoint
    # For this demo, we'll initialize without pre-trained weights
    model_path = None  # Set to "../../backend/model_checkpoints/best_model.pth" after training
    
    print("\n📁 Setup:")
    print(f"  Data directory: {data_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Model: {'Untrained (random weights)' if model_path is None else model_path}")
    
    # Find first event with images
    events = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    if not events:
        print("\n❌ No events found in data directory!")
        return
    
    # Process first event
    event_dir = events[0]
    images = sorted([str(p) for p in event_dir.glob("*.jpg")])
    
    if len(images) < 2:
        print(f"\n❌ Event {event_dir.name} has insufficient images!")
        return
    
    print(f"\n🎬 Processing Event: {event_dir.name}")
    print(f"  Images: {len(images)}")
    
    # ==================== Stage 1: Motion Detection ====================
    print("\n" + "=" * 80)
    print("STAGE 1: Motion Detection")
    print("=" * 80)
    
    motion_detector = MotionDetector(threshold=25, min_area=500)
    
    print("\n⏳ Detecting motion...")
    motion_result = motion_detector.detect_motion(images)
    
    print(f"\n✓ Motion Detection Complete!")
    print(f"  Motion Detected: {motion_result.motion_detected}")
    print(f"  Total Motion Area: {motion_result.total_motion_area:,} pixels")
    print(f"  Confidence: {motion_result.confidence:.2%}")
    print(f"  Regions Found: {len(motion_result.motion_regions)}")
    
    if motion_result.motion_regions:
        print("\n  Top 3 Regions:")
        for i, region in enumerate(motion_result.motion_regions[:3], 1):
            print(f"    {i}. Position: ({region.x}, {region.y}), "
                  f"Size: {region.width}x{region.height}, "
                  f"Area: {region.area:,}px")
    
    # Save motion visualization
    motion_vis_path = output_dir / f"{event_dir.name}_motion.jpg"
    middle_idx = len(images) // 2
    motion_detector.visualize_motion(
        images[middle_idx],
        motion_result,
        output_path=str(motion_vis_path),
        show_mask=True,
        show_boxes=True
    )
    print(f"\n📸 Motion visualization saved: {motion_vis_path}")
    
    # ==================== Stage 2: Relevance Prediction ====================
    print("\n" + "=" * 80)
    print("STAGE 2: Relevance Mask Prediction")
    print("=" * 80)
    
    if not motion_result.motion_detected:
        print("\n⚠️  No motion detected - skipping relevance prediction")
        return
    
    # Initialize predictor
    print("\n⏳ Initializing relevance predictor...")
    
    try:
        predictor = RelevanceMaskPredictor(
            model_path=model_path,
            relevance_threshold=0.5
        )
        print("✓ Predictor initialized")
        
        if model_path is None:
            print("\n⚠️  WARNING: Using untrained model (random weights)")
            print("   Results will be random! Train model first:")
            print("   python backend/train_relevance_model.py --phase both")
    
    except Exception as e:
        print(f"\n❌ Failed to initialize predictor: {e}")
        return
    
    # Predict relevance
    print("\n⏳ Predicting relevance...")
    middle_frame = images[middle_idx]
    
    relevance_result = predictor.predict(middle_frame, motion_result)
    
    print(f"\n✓ Relevance Prediction Complete!")
    print(f"  Processing Time: {relevance_result.processing_time_ms:.1f}ms")
    print(f"  Relevant Regions: {relevance_result.num_relevant}")
    print(f"  Irrelevant Regions: {relevance_result.num_irrelevant}")
    
    if relevance_result.relevant_regions:
        print("\n  Relevant Regions:")
        for i, region in enumerate(relevance_result.relevant_regions, 1):
            print(f"    {i}. Position: ({region.x}, {region.y}), "
                  f"Relevance Score: {region.relevance_score:.2f}")
    
    if relevance_result.irrelevant_regions:
        print("\n  Irrelevant Regions:")
        for i, region in enumerate(relevance_result.irrelevant_regions, 1):
            print(f"    {i}. Position: ({region.x}, {region.y}), "
                  f"Relevance Score: {region.relevance_score:.2f}")
    
    # Save relevance visualization
    relevance_vis_path = output_dir / f"{event_dir.name}_relevance.jpg"
    frame = cv2.imread(middle_frame)
    predictor.visualize(frame, relevance_result, output_path=str(relevance_vis_path))
    print(f"\n📸 Relevance visualization saved: {relevance_vis_path}")
    
    # Save JSON results
    json_path = output_dir / f"{event_dir.name}_results.json"
    results = {
        "event_id": event_dir.name,
        "timestamp": datetime.now().isoformat(),
        "motion_detection": {
            "motion_detected": motion_result.motion_detected,
            "total_area": motion_result.total_motion_area,
            "confidence": float(motion_result.confidence),
            "num_regions": len(motion_result.motion_regions)
        },
        "relevance_prediction": relevance_result.to_dict()
    }
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved: {json_path}")
    
    # ==================== Decision ====================
    print("\n" + "=" * 80)
    print("FINAL DECISION")
    print("=" * 80)
    
    if not motion_result.motion_detected:
        print("\n❌ NO ALERT: No motion detected")
    elif relevance_result.num_relevant == 0:
        print("\n❌ NO ALERT: Motion detected but not relevant")
        print("   (Likely false alarm: trees, shadows, etc.)")
    else:
        print("\n✅ ALERT TRIGGERED!")
        print(f"   {relevance_result.num_relevant} relevant motion region(s) detected")
        max_score = max(r.relevance_score for r in relevance_result.relevant_regions)
        print(f"   Confidence: {max_score:.2%}")
        print(f"   Action: Send notification / Record event")
    
    print("\n" + "=" * 80)
    print("✓ Pipeline Complete!")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - Motion visualization: {motion_vis_path.name}")
    print(f"  - Relevance visualization: {relevance_vis_path.name}")
    print(f"  - JSON results: {json_path.name}")


if __name__ == "__main__":
    main()
