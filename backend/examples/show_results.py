"""
Display relevance mask visualization results
"""

import cv2
import numpy as np
from pathlib import Path

# Paths
output_dir = Path("../../data/relevance_outputs")
motion_img_path = output_dir / "20251222_032856_motion.jpg"
relevance_img_path = output_dir / "20251222_032856_relevance.jpg"

print("=" * 80)
print("📊 Visualization Results")
print("=" * 80)

# Load images
if motion_img_path.exists():
    motion_img = cv2.imread(str(motion_img_path))
    print(f"\n✓ Motion Detection Visualization:")
    print(f"  File: {motion_img_path.name}")
    print(f"  Size: {motion_img.shape[1]}x{motion_img.shape[0]} pixels")
    print(f"  Legend:")
    print(f"    🟢 Green boxes = Motion detected regions")
    print(f"    🔵 Blue dots = Region centroids")
    print(f"    🟢 Green overlay = Binary motion mask")
else:
    print(f"\n❌ Motion image not found: {motion_img_path}")

if relevance_img_path.exists():
    relevance_img = cv2.imread(str(relevance_img_path))
    print(f"\n✓ Relevance Mask Visualization:")
    print(f"  File: {relevance_img_path.name}")
    print(f"  Size: {relevance_img.shape[1]}x{relevance_img.shape[0]} pixels")
    print(f"  Legend:")
    print(f"    🟢 Green boxes = RELEVANT motion (real alerts)")
    print(f"    🔴 Red boxes = IRRELEVANT motion (false alarms)")
    print(f"    🟢 Green overlay = Relevance mask (relevant areas)")
    print(f"    Labels show relevance scores (0-1)")
else:
    print(f"\n❌ Relevance image not found: {relevance_img_path}")

# Create side-by-side comparison
if motion_img_path.exists() and relevance_img_path.exists():
    # Resize to same height
    h1, w1 = motion_img.shape[:2]
    h2, w2 = relevance_img.shape[:2]
    
    target_height = 480
    scale1 = target_height / h1
    scale2 = target_height / h2
    
    motion_resized = cv2.resize(motion_img, (int(w1 * scale1), target_height))
    relevance_resized = cv2.resize(relevance_img, (int(w2 * scale2), target_height))
    
    # Add text labels
    label_height = 40
    motion_labeled = np.zeros((target_height + label_height, motion_resized.shape[1], 3), dtype=np.uint8)
    relevance_labeled = np.zeros((target_height + label_height, relevance_resized.shape[1], 3), dtype=np.uint8)
    
    motion_labeled[label_height:] = motion_resized
    relevance_labeled[label_height:] = relevance_resized
    
    # Add labels
    cv2.putText(motion_labeled, "STAGE 1: Motion Detection", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(relevance_labeled, "STAGE 2: Relevance Filtering", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Combine side by side
    comparison = np.hstack([motion_labeled, relevance_labeled])
    
    # Save comparison
    comparison_path = output_dir / "comparison.jpg"
    cv2.imwrite(str(comparison_path), comparison)
    
    print(f"\n✓ Created side-by-side comparison:")
    print(f"  File: {comparison_path.name}")
    print(f"  Size: {comparison.shape[1]}x{comparison.shape[0]} pixels")

print("\n" + "=" * 80)
print("📍 Image Locations:")
print("=" * 80)
print(f"\nAll images saved in: {output_dir.absolute()}")
print(f"\n1. Motion Detection:")
print(f"   {motion_img_path.absolute()}")
print(f"\n2. Relevance Filtering:")
print(f"   {relevance_img_path.absolute()}")
print(f"\n3. Side-by-Side Comparison:")
print(f"   {output_dir.absolute() / 'comparison.jpg'}")

print("\n" + "=" * 80)
print("💡 How to View:")
print("=" * 80)
print(f"\nOpen with image viewer:")
print(f"  xdg-open {relevance_img_path.absolute()}")
print(f"\nOr in VS Code:")
print(f"  Click on the file in the file explorer")
print(f"\nOr with any image viewer of your choice")
print("=" * 80)
