"""
Pipeline Visualizer: Generate annotated images for each stage transition.

This module extends the pipeline to create visual outputs showing:
- Motion detection boxes
- Stage information
- Decision details
- Scores and metrics
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import cv2


class PipelineVisualizer:
    """Generates annotated images for pipeline stages."""
    
    def __init__(self, output_dir: str = "data/pipeline_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def draw_boxes(self, image: np.ndarray, boxes: List[Tuple[int, int, int, int]], 
                   color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """Draw bounding boxes on image."""
        img = image.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return img
    
    def add_text_overlay(self, image: np.ndarray, lines: List[str], 
                        bg_color: Tuple[int, int, int] = (0, 0, 0),
                        text_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """Add multi-line text overlay to image."""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Calculate text box size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 20
        padding = 10
        
        # Draw background rectangle
        box_height = len(lines) * line_height + 2 * padding
        cv2.rectangle(img, (0, 0), (w, box_height), bg_color, -1)
        
        # Draw text lines
        y_offset = padding + 15
        for line in lines:
            cv2.putText(img, line, (padding, y_offset), font, font_scale, 
                       text_color, thickness, cv2.LINE_AA)
            y_offset += line_height
        
        return img
    
    def visualize_stage_3(self, image: np.ndarray, motion_boxes: List[Tuple[int, int, int, int]],
                         stage3_result: Dict, camera_id: str, alert_id: str) -> Path:
        """Generate visualization for Stage 3 (HeatMap Filter)."""
        img = self.draw_boxes(image, motion_boxes, color=(0, 255, 255), thickness=2)
        
        heat_zone = stage3_result.get("heat_zone", "unknown")
        anomaly_score = stage3_result.get("anomaly_score", 0.0)
        next_stage = stage3_result.get("next_stage", 4)
        
        # Color code by heat zone
        if heat_zone == "cold":
            zone_color = (255, 100, 100)  # Blue-ish for cold
            zone_text = "COLD (Anomalous)"
        else:
            zone_color = (100, 100, 255)  # Red-ish for hot
            zone_text = "HOT (Normal)"
        
        lines = [
            f"Camera: {camera_id} | Alert: {alert_id}",
            f"STAGE 3: HeatMap-Based Filter",
            f"Heat Zone: {zone_text}",
            f"Anomaly Score: {anomaly_score:.3f}",
            f"Decision: Route to Stage {next_stage}",
            f"Motion Boxes: {len(motion_boxes)}"
        ]
        
        img = self.add_text_overlay(img, lines, bg_color=(0, 0, 50), text_color=(255, 255, 255))
        
        # Save image
        output_path = self.output_dir / f"{alert_id}_stage3.jpg"
        cv2.imwrite(str(output_path), img)
        return output_path
    
    def visualize_stage_4(self, image: np.ndarray, motion_boxes: List[Tuple[int, int, int, int]],
                         stage4_result: Dict, camera_id: str, alert_id: str) -> Path:
        """Generate visualization for Stage 4 (Memory-Based Anomaly Detection)."""
        img = self.draw_boxes(image, motion_boxes, color=(255, 0, 255), thickness=2)
        
        event_sim = stage4_result.get("event_sim", 0.0)
        support = stage4_result.get("event_support", 0)
        decision = stage4_result.get("decision", "PASS")
        
        # Color code by decision
        if decision == "FILTER":
            decision_color = (0, 0, 255)  # Red for escalate
            decision_text = "FILTER (Escalate)"
        elif decision == "DROP":
            decision_color = (0, 255, 0)  # Green for drop
            decision_text = "DROP (Reject)"
        else:
            decision_color = (0, 165, 255)  # Orange for uncertain
            decision_text = "PASS (Uncertain)"
        
        lines = [
            f"Camera: {camera_id} | Alert: {alert_id}",
            f"STAGE 4: Memory-Based Anomaly Detection",
            f"Decision: {decision_text}",
            f"Event Similarity: {event_sim:.3f}",
            f"Support Count: {support} similar events",
            f"Motion Boxes: {len(motion_boxes)}"
        ]
        
        img = self.add_text_overlay(img, lines, bg_color=(50, 0, 50), text_color=(255, 255, 255))
        
        # Save image
        output_path = self.output_dir / f"{alert_id}_stage4.jpg"
        cv2.imwrite(str(output_path), img)
        return output_path
    
    def visualize_stage_5(self, image: np.ndarray, stage5_result: Dict,
                         camera_id: str, alert_id: str) -> Path:
        """Generate visualization for Stage 5 (YOLO Object Detection)."""
        img = image.copy()
        
        if stage5_result and "detections" in stage5_result:
            detections = stage5_result["detections"]
            
            # Draw YOLO detections
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                conf = det["confidence"]
                cls = det["class"]
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cls}: {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)
            
            detected_classes = [d["class"] for d in detections]
            class_counts = {}
            for cls in detected_classes:
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            class_summary = ", ".join([f"{cls}({cnt})" for cls, cnt in class_counts.items()])
        else:
            class_summary = "No objects detected"
        
        lines = [
            f"Camera: {camera_id} | Alert: {alert_id}",
            f"STAGE 5: YOLO Object Detection",
            f"Detections: {class_summary}",
            f"Total Objects: {len(detections) if stage5_result and 'detections' in stage5_result else 0}"
        ]
        
        img = self.add_text_overlay(img, lines, bg_color=(0, 50, 0), text_color=(255, 255, 255))
        
        # Save image
        output_path = self.output_dir / f"{alert_id}_stage5.jpg"
        cv2.imwrite(str(output_path), img)
        return output_path
    
    def visualize_final_decision(self, image: np.ndarray, pipeline_result: Dict,
                                motion_boxes: List[Tuple[int, int, int, int]],
                                camera_id: str, alert_id: str) -> Path:
        """Generate final decision visualization with all stages."""
        img = self.draw_boxes(image, motion_boxes, color=(255, 255, 0), thickness=3)
        
        final_decision = pipeline_result.get("final_decision", "UNKNOWN")
        explanation = pipeline_result.get("explanation", "")
        
        # Color code by final decision
        if final_decision == "ESCALATE":
            decision_color = (0, 0, 255)  # Red
            bg_color = (50, 0, 0)
        else:
            decision_color = (0, 255, 0)  # Green
            bg_color = (0, 50, 0)
        
        # Wrap explanation text
        max_line_length = 80
        explanation_lines = []
        words = explanation.split()
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= max_line_length:
                current_line += (word + " ")
            else:
                explanation_lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            explanation_lines.append(current_line.strip())
        
        lines = [
            f"Camera: {camera_id} | Alert: {alert_id}",
            f"FINAL DECISION: {final_decision}",
            "=" * 60,
        ] + explanation_lines
        
        img = self.add_text_overlay(img, lines, bg_color=bg_color, text_color=(255, 255, 255))
        
        # Save image
        output_path = self.output_dir / f"{alert_id}_final.jpg"
        cv2.imwrite(str(output_path), img)
        return output_path
