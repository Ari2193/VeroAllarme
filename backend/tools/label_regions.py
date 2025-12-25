"""
Interactive Tool for Manual Region Labeling

This tool allows you to manually label regions in camera frames as relevant/irrelevant.
The labels will be used to train a supervised model.
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict

class RegionLabeler:
    """Interactive tool for labeling relevant regions"""
    
    def __init__(self, camera_name: str, output_dir: str = "data/manual_labels"):
        self.camera_name = camera_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.labels_file = self.output_dir / f"{camera_name}_labels.json"
        self.labels = self._load_labels()
        
        # Drawing state
        self.drawing = False
        self.current_box = None
        self.boxes = []
        self.current_label = "relevant"  # or "irrelevant"
        
        print("=" * 80)
        print(f"🏷️  Region Labeler for {camera_name}")
        print("=" * 80)
        print("\nInstructions:")
        print("  - Click and drag to draw a box")
        print("  - Press 'r' to mark next box as RELEVANT (green)")
        print("  - Press 'i' to mark next box as IRRELEVANT (red)")
        print("  - Press 'u' to undo last box")
        print("  - Press 's' to save and continue")
        print("  - Press 'q' to quit without saving")
        print("  - Press SPACE to skip current frame")
        print("=" * 80)
    
    def _load_labels(self) -> Dict:
        """Load existing labels from file"""
        if self.labels_file.exists():
            with open(self.labels_file, 'r') as f:
                return json.load(f)
        return {"camera": self.camera_name, "frames": {}}
    
    def _save_labels(self):
        """Save labels to file"""
        with open(self.labels_file, 'w') as f:
            json.dump(self.labels, f, indent=2)
        print(f"✓ Saved labels to {self.labels_file}")
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing boxes"""
        frame, display = param
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_box = [(x, y), (x, y)]
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box[1] = (x, y)
                # Redraw
                display_copy = display.copy()
                self._draw_current_box(display_copy)
                cv2.imshow("Label Regions", display_copy)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.current_box[1] = (x, y)
                
                # Add box
                x1, y1 = self.current_box[0]
                x2, y2 = self.current_box[1]
                
                # Normalize coordinates
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Ensure minimum size
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    self.boxes.append({
                        "bbox": [x1, y1, x2, y2],
                        "label": self.current_label
                    })
                
                self.current_box = None
    
    def _draw_current_box(self, display):
        """Draw the box being drawn"""
        if self.current_box:
            color = (0, 255, 0) if self.current_label == "relevant" else (0, 0, 255)
            cv2.rectangle(display, self.current_box[0], self.current_box[1], color, 2)
    
    def _draw_boxes(self, display):
        """Draw all labeled boxes"""
        for box in self.boxes:
            x1, y1, x2, y2 = box["bbox"]
            color = (0, 255, 0) if box["label"] == "relevant" else (0, 0, 255)
            label_text = "REL" if box["label"] == "relevant" else "IRR"
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def label_frame(self, frame_path: str, frame: np.ndarray) -> bool:
        """
        Label a single frame.
        Returns True if saved, False if skipped/quit
        """
        self.boxes = []
        
        # Check if already labeled
        if frame_path in self.labels["frames"]:
            print(f"Frame {frame_path} already labeled, skipping...")
            return True
        
        h, w = frame.shape[:2]
        display = frame.copy()
        
        # Add instructions overlay
        cv2.putText(display, f"Mode: {self.current_label.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "Press 'r'=relevant, 'i'=irrelevant, 's'=save, 'q'=quit, SPACE=skip",
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show window
        cv2.namedWindow("Label Regions")
        cv2.setMouseCallback("Label Regions", self._mouse_callback, (frame, display))
        
        while True:
            display_copy = display.copy()
            self._draw_boxes(display_copy)
            if self.current_box:
                self._draw_current_box(display_copy)
            
            # Update mode display
            cv2.putText(display_copy, f"Mode: {self.current_label.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 0) if self.current_label == "relevant" else (0, 0, 255), 2)
            
            cv2.imshow("Label Regions", display_copy)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):
                self.current_label = "relevant"
                print("Mode: RELEVANT (green)")
            
            elif key == ord('i'):
                self.current_label = "irrelevant"
                print("Mode: IRRELEVANT (red)")
            
            elif key == ord('u'):
                if self.boxes:
                    removed = self.boxes.pop()
                    print(f"Removed last box: {removed['label']}")
            
            elif key == ord('s'):
                # Save
                if len(self.boxes) > 0:
                    self.labels["frames"][frame_path] = {
                        "width": w,
                        "height": h,
                        "regions": self.boxes
                    }
                    self._save_labels()
                    print(f"✓ Saved {len(self.boxes)} regions")
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("⚠ No regions drawn. Draw at least one region or press SPACE to skip.")
            
            elif key == ord(' '):
                # Skip
                print("Skipped frame")
                cv2.destroyAllWindows()
                return True
            
            elif key == ord('q'):
                # Quit
                cv2.destroyAllWindows()
                return False
        
        return False
    
    def label_directory(self, data_dir: str, max_frames: int = 10):
        """Label frames from directory"""
        data_path = Path(data_dir)
        
        # Get event directories
        event_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
        
        labeled_count = 0
        
        for event_dir in event_dirs:
            if labeled_count >= max_frames:
                break
            
            images = sorted(event_dir.glob("*.jpg"))
            if len(images) == 0:
                continue
            
            # Use middle frame
            middle_idx = len(images) // 2
            frame_path = str(images[middle_idx].relative_to(data_path.parent))
            
            # Skip if already labeled
            if frame_path in self.labels["frames"]:
                continue
            
            print(f"\nLabeling frame {labeled_count + 1}/{max_frames}")
            print(f"Event: {event_dir.name}")
            
            frame = cv2.imread(str(images[middle_idx]))
            
            if not self.label_frame(frame_path, frame):
                # User quit
                print("\n" + "=" * 80)
                print(f"Labeled {labeled_count} frames total")
                print("=" * 80)
                break
            
            labeled_count += 1
        
        print("\n" + "=" * 80)
        print(f"✅ Labeling complete! Labeled {labeled_count} frames")
        print(f"Labels saved to: {self.labels_file}")
        print("=" * 80)
        
        return labeled_count


def main():
    """Run interactive labeling"""
    import sys
    
    camera_name = "Factory"
    data_dir = "../data/training/camera-events/Factory"
    max_frames = 20  # Label 20 frames
    
    labeler = RegionLabeler(camera_name)
    labeler.label_directory(data_dir, max_frames=max_frames)


if __name__ == "__main__":
    main()
