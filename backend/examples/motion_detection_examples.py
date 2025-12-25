"""
Integration example and usage demonstration for motion detection
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.motion_detection import MotionDetector, detect_motion_from_paths
import json


def example_basic_usage():
    """Basic usage example"""
    print("=" * 60)
    print("Example 1: Basic Motion Detection")
    print("=" * 60)
    
    # Define image paths
    image_paths = [
        "data/training/camera-events/Factory/20251222_032856/0.jpg",
        "data/training/camera-events/Factory/20251222_032856/1.jpg",
        "data/training/camera-events/Factory/20251222_032856/2.jpg"
    ]
    
    # Check if files exist
    if not all(Path(p).exists() for p in image_paths):
        print("⚠️  Training images not found. Using placeholder paths.")
        return
    
    # Detect motion
    result = detect_motion_from_paths(image_paths, threshold=25, min_area=500)
    
    # Print results
    print(f"\n✓ Motion Detected: {result.motion_detected}")
    print(f"  Total Motion Area: {result.total_motion_area} pixels")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Number of Regions: {len(result.motion_regions)}")
    
    # Print each region
    for i, region in enumerate(result.motion_regions, 1):
        print(f"\n  Region #{i}:")
        print(f"    Position: ({region.x}, {region.y})")
        print(f"    Size: {region.width}x{region.height}")
        print(f"    Area: {region.area} pixels")
        print(f"    Centroid: {region.centroid}")


def example_advanced_usage():
    """Advanced usage with custom parameters and visualization"""
    print("\n" + "=" * 60)
    print("Example 2: Advanced Motion Detection with Visualization")
    print("=" * 60)
    
    # Define image paths
    image_paths = [
        "data/training/camera-events/Field/20251222_040315/0.jpg",
        "data/training/camera-events/Field/20251222_040315/1.jpg",
        "data/training/camera-events/Field/20251222_040315/2.jpg"
    ]
    
    # Check if files exist
    if not all(Path(p).exists() for p in image_paths):
        print("⚠️  Training images not found.")
        return
    
    # Create detector with custom parameters
    detector = MotionDetector(
        threshold=30,           # Higher threshold = less sensitive
        min_area=1000,          # Larger minimum area
        blur_kernel=(25, 25),   # More aggressive blur
        morph_kernel_size=7     # Larger morphological kernel
    )
    
    # Detect motion
    result = detector.detect_motion(image_paths)
    
    # Create visualization
    output_path = "data/motion_visualization.jpg"
    detector.visualize_motion(
        image_paths[1],  # Use middle frame
        result,
        output_path=output_path,
        show_mask=True,
        show_boxes=True,
        show_centroids=True
    )
    
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Export to JSON
    result_dict = result.to_dict()
    json_path = "data/motion_result.json"
    with open(json_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    print(f"✓ JSON result saved to: {json_path}")


def example_batch_processing():
    """Process multiple events in batch"""
    print("\n" + "=" * 60)
    print("Example 3: Batch Processing Multiple Events")
    print("=" * 60)
    
    # Find all event directories
    base_path = Path("data/training/camera-events/Factory")
    
    if not base_path.exists():
        print("⚠️  Training data not found.")
        return
    
    event_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])[:5]
    
    detector = MotionDetector(threshold=25, min_area=500)
    
    results = []
    
    for event_dir in event_dirs:
        images = sorted([str(p) for p in event_dir.glob("*.jpg")])
        
        if len(images) < 2:
            continue
        
        try:
            result = detector.detect_motion(images)
            results.append({
                "event": event_dir.name,
                "motion_detected": result.motion_detected,
                "num_regions": len(result.motion_regions),
                "total_area": result.total_motion_area,
                "confidence": result.confidence
            })
        except Exception as e:
            print(f"❌ Error processing {event_dir.name}: {e}")
    
    # Print summary
    print(f"\n✓ Processed {len(results)} events")
    print(f"  Motion detected: {sum(r['motion_detected'] for r in results)}")
    print(f"  Average confidence: {sum(r['confidence'] for r in results) / len(results):.2%}")
    
    # Print details
    print("\n  Event Details:")
    for r in results:
        status = "✓" if r['motion_detected'] else "✗"
        print(f"    {status} {r['event']}: {r['num_regions']} regions, {r['confidence']:.1%} confidence")


def example_api_integration():
    """Example of how to integrate with FastAPI"""
    print("\n" + "=" * 60)
    print("Example 4: API Integration Format")
    print("=" * 60)
    
    # Simulate API request
    image_paths = [
        "data/training/camera-events/Trees/20251222_040741/0.jpg",
        "data/training/camera-events/Trees/20251222_040741/1.jpg",
        "data/training/camera-events/Trees/20251222_040741/2.jpg"
    ]
    
    if not all(Path(p).exists() for p in image_paths):
        print("⚠️  Training images not found.")
        return
    
    # Process
    result = detect_motion_from_paths(image_paths)
    
    # Format for API response
    api_response = {
        "status": "success",
        "data": result.to_dict(),
        "metadata": {
            "algorithm": "frame_differencing",
            "version": "1.0.0",
            "processing_time_ms": 45  # Placeholder
        }
    }
    
    print("\nAPI Response Format:")
    print(json.dumps(api_response, indent=2))


if __name__ == "__main__":
    print("\n🔍 Motion Detection Examples\n")
    
    try:
        example_basic_usage()
        example_advanced_usage()
        example_batch_processing()
        example_api_integration()
        
        print("\n" + "=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
