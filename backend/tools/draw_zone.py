"""
Draw a single relevant zone polygon for the camera

This is simpler - just draw ONE polygon that defines the relevant area.
Everything inside = relevant, everything outside = irrelevant.
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple

class ZoneDrawer:
    """Draw a single polygon zone for relevant area"""
    
    def __init__(self, camera_name: str, output_dir: str = "data/zones"):
        self.camera_name = camera_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.zone_file = self.output_dir / f"{camera_name}_zone.json"
        
        self.points = []
        self.frame_size = None
        
        print("=" * 80)
        print(f"🎯 Zone Drawer for {camera_name}")
        print("=" * 80)
        print("\nInstructions:")
        print("  - Click to add points to the polygon")
        print("  - Press 'u' to undo last point")
        print("  - Press 'c' to clear all points")
        print("  - Press 's' to save zone")
        print("  - Press 'q' to quit without saving")
        print("\nTip: Draw a polygon around the RELEVANT area")
        print("     (e.g., building entrance, parking area, etc.)")
        print("=" * 80)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"Added point {len(self.points)}: ({x}, {y})")
    
    def draw_zone(self, frame: np.ndarray) -> bool:
        """
        Draw zone on frame.
        Returns True if saved, False if quit
        """
        self.frame_size = (frame.shape[1], frame.shape[0])
        
        cv2.namedWindow("Draw Zone")
        cv2.setMouseCallback("Draw Zone", self._mouse_callback)
        
        while True:
            display = frame.copy()
            
            # Draw points
            for i, pt in enumerate(self.points):
                cv2.circle(display, pt, 5, (0, 255, 0), -1)
                cv2.putText(display, str(i + 1), (pt[0] + 10, pt[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw lines
            if len(self.points) > 1:
                for i in range(len(self.points) - 1):
                    cv2.line(display, self.points[i], self.points[i + 1], (0, 255, 0), 2)
            
            # Draw closing line
            if len(self.points) > 2:
                cv2.line(display, self.points[-1], self.points[0], (0, 255, 0), 2)
                
                # Fill polygon semi-transparent
                overlay = display.copy()
                pts = np.array(self.points, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
            
            # Instructions
            h, w = frame.shape[:2]
            cv2.putText(display, f"Points: {len(self.points)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "Click to add point | 'u'=undo | 's'=save | 'q'=quit",
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Draw Zone", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('u'):
                if self.points:
                    removed = self.points.pop()
                    print(f"Removed point: {removed}")
            
            elif key == ord('c'):
                self.points = []
                print("Cleared all points")
            
            elif key == ord('s'):
                if len(self.points) >= 3:
                    # Save
                    zone_data = {
                        "camera": self.camera_name,
                        "frame_size": {"width": self.frame_size[0], "height": self.frame_size[1]},
                        "polygon": self.points
                    }
                    
                    with open(self.zone_file, 'w') as f:
                        json.dump(zone_data, f, indent=2)
                    
                    print(f"\n✓ Saved zone with {len(self.points)} points to {self.zone_file}")
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("⚠ Need at least 3 points to form a polygon")
            
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False
        
        return False


def load_zone(camera_name: str, zone_dir: str = "data/zones") -> dict:
    """Load saved zone"""
    zone_file = Path(zone_dir) / f"{camera_name}_zone.json"
    
    if zone_file.exists():
        with open(zone_file, 'r') as f:
            return json.load(f)
    
    return None


def point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
    """Check if point is inside polygon using ray casting"""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def visualize_zone(frame: np.ndarray, zone_data: dict, alpha: float = 0.4) -> np.ndarray:
    """Visualize zone on frame"""
    display = frame.copy()
    
    if zone_data and "polygon" in zone_data:
        polygon = zone_data["polygon"]
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        
        # Fill relevant area (green)
        overlay = display.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        display = cv2.addWeighted(display, 1 - alpha, overlay, alpha, 0)
        
        # Draw border
        cv2.polylines(display, [pts], True, (0, 255, 0), 3)
        
        # Add label
        cv2.putText(display, "RELEVANT ZONE", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return display


def main():
    """Interactive zone drawing"""
    camera_name = "Factory"
    data_dir = Path("../data/training/camera-events/Factory")
    
    # Load a sample frame
    event_dir = next(d for d in data_dir.iterdir() if d.is_dir())
    images = sorted(event_dir.glob("*.jpg"))
    frame = cv2.imread(str(images[0]))
    
    print(f"\nFrame size: {frame.shape[1]}x{frame.shape[0]}")
    print("\nDraw a polygon around the RELEVANT area")
    print("(e.g., building entrance, parking spots, etc.)\n")
    
    drawer = ZoneDrawer(camera_name)
    
    if drawer.draw_zone(frame):
        # Show result
        zone_data = load_zone(camera_name)
        result = visualize_zone(frame, zone_data)
        
        output_path = Path("data/relevance_outputs/zone_visualization.jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)
        
        print(f"✓ Saved visualization to {output_path}")
        
        # Test a point
        test_x, test_y = frame.shape[1] // 2, frame.shape[0] // 2
        is_inside = point_in_polygon((test_x, test_y), zone_data["polygon"])
        print(f"\nTest: Center point ({test_x}, {test_y}) is {'INSIDE' if is_inside else 'OUTSIDE'} zone")


if __name__ == "__main__":
    main()
