#!/usr/bin/env python3
"""
Quick Start: Alert Processing Pipeline

This example shows how to use the 3-stage filtering pipeline
to process security alerts and make ESCALATE/DROP decisions.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import importlib.util

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Handle hyphenated import for filter-heat-map
filter_heat_map_path = backend_path / "core" / "filter-heat-map"
spec = importlib.util.spec_from_file_location(
    "filter_heat_map",
    filter_heat_map_path / "__init__.py"
)
filter_heat_map_module = importlib.util.module_from_spec(spec)
sys.modules["filter_heat_map"] = filter_heat_map_module
spec.loader.exec_module(filter_heat_map_module)

# Handle hyphenated import for compare-events
compare_events_path = backend_path / "core" / "compare-events"
spec2 = importlib.util.spec_from_file_location(
    "compare_events",
    compare_events_path / "__init__.py"
)
compare_events_module = importlib.util.module_from_spec(spec2)
sys.modules["compare_events"] = compare_events_module
spec2.loader.exec_module(compare_events_module)

from filter_heat_map import FilterHeatMap
from compare_events import AnomalyDetector
from core.pipeline import create_pipeline


def create_dummy_alert_data():
    """Generate synthetic alert data for demonstration."""
    h, w = 480, 640
    
    # Three consecutive frames (I1, I2, I3)
    images = [np.random.randint(50, 200, (h, w, 3), dtype=np.uint8) for _ in range(3)]
    
    # Motion mask (binary)
    motion_mask = np.zeros((h, w), dtype=np.uint8)
    motion_mask[100:200, 150:250] = 255  # Blob in quiet region
    
    # Motion bounding boxes (x1, y1, x2, y2)
    motion_boxes = [(150, 100, 250, 200)]
    
    return images, motion_mask, motion_boxes


def main():
    print("=" * 70)
    print("ALERT PROCESSING PIPELINE - Quick Start")
    print("=" * 70)
    
    # Step 1: Initialize components
    print("\n[1] Initializing pipeline components...")
    
    heatmap_filter = FilterHeatMap(
        camera_id="front_entrance",
        storage_path="data/heatmaps",
        frame_shape=(480, 640),
        history_days=30,
        decay_factor=0.95,
        anomaly_threshold=0.5,
    )
    print("   ✓ Heat Map Filter (Stage 3)")
    
    # NOTE: Stage 4 requires FAISS and transformers
    # For demo, using stub detector
    anomaly_detector = AnomalyDetector(
        embedding_model="clip",       # CLIP embeddings (or "dinov2")
        sim_strong=0.92,              # Strong similarity threshold
        sim_weak=0.85,                # Weak similarity threshold
        support_min=8,                # Min supporting past events
    )
    print("   ✓ Anomaly Detector (Stage 4)")
    
    # Create pipeline
    pipeline = create_pipeline(
        heatmap_filter=heatmap_filter,
        anomaly_detector=anomaly_detector,
        yolo_detector=None,  # Optional Stage 5
    )
    print("   ✓ Pipeline orchestrator created")
    
    # Step 2: Generate alert data
    print("\n[2] Generating synthetic alert...")
    images, motion_mask, motion_boxes = create_dummy_alert_data()
    print(f"   ✓ 3 frames: {images[0].shape}")
    print(f"   ✓ Motion mask shape: {motion_mask.shape}")
    print(f"   ✓ Motion boxes: {motion_boxes}")
    
    # Step 3: Process alert
    print("\n[3] Processing alert through pipeline...")
    result = pipeline.process_alert(
        camera_id="front_entrance",
        alert_id="ALERT_20250101_120000_001",
        images=images,
        motion_mask=motion_mask,
        motion_boxes=motion_boxes,
    )
    
    # Step 4: Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nAlert ID:       {result['alert_id']}")
    print(f"Camera:         {result['camera_id']}")
    print(f"Timestamp:      {result['timestamp']}")
    print(f"\n{'FINAL DECISION:':<20} {result['final_decision']}")
    print(f"\n{'Explanation:':<20}")
    for line in result['explanation'].split('.'):
        if line.strip():
            print(f"  • {line.strip()}.")
    
    # Stage 3 details
    print(f"\n[Stage 3] Heat Map Analysis:")
    s3 = result['stage_3_result']
    if 'error' not in s3:
        print(f"  Anomaly Score: {s3['anomaly_score']:.3f}")
        print(f"  Heat Zone:     {s3['heat_zone'].upper()}")
        print(f"  Next Stage:    {s3['next_stage']}")
        stats = s3.get('stats', {})
        if stats:
            print(f"  Total Events:  {stats.get('total_events', '?')}")
            print(f"  Active Pixels: {stats.get('active_pixels', '?')}")
    else:
        print(f"  Error: {s3['error']}")
    
    # Stage 4 details
    print(f"\n[Stage 4] Memory-based Detection:")
    s4 = result['stage_4_result']
    if 'error' not in s4:
        print(f"  Similarity:    {s4['event_sim']:.3f}")
        print(f"  Support:       {s4['event_support']} past events")
        print(f"  Decision:      {s4['decision']}")
        if s4.get('neighbors'):
            print(f"  Neighbors:     {len(s4['neighbors'])} similar events found")
    else:
        print(f"  Error: {s4['error']}")
    
    # Decision explanation
    print(f"\n[Decision Logic]:")
    if result['final_decision'] == "ESCALATE":
        print("  ✓ Alert escalated to human review / Stage 5 detection")
    else:
        print("  ✗ Alert filtered out (likely false alarm)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
