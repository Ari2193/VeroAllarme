"""
Alert Processing Pipeline: Stages 3, 4, and 5 integration.

Flow:
1. Stage 3 (HeatMap) → heat_zone (cold/hot), next_stage decision
2. Stage 4 (Memory) → event_sim, support, final decision (FILTER/DROP/PASS)
3. Stage 5 (YOLO) → object detection if routed from Stage 4

Usage:
    from backend.core.pipeline import create_pipeline
    from backend.core.filter_heat_map import FilterHeatMap
    from backend.core.compare_events import AnomalyDetector
    
    pipeline = create_pipeline(
        heatmap_filter=FilterHeatMap(),
        anomaly_detector=AnomalyDetector()
    )
    
    result = pipeline.process_alert(
        camera_id="camera_1",
        alert_id="alert_001",
        images=images,
        motion_mask=mask,
        motion_boxes=boxes
    )
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np


class AlertProcessingPipeline:
    """Orchestrates stages 3, 4, and 5 for alert filtering."""
    
    def __init__(
        self,
        heatmap_filter,  # FilterHeatMap instance from Stage 3
        anomaly_detector,  # AnomalyDetector instance from Stage 4
        yolo_detector=None,  # YOLOv8 detector for Stage 5 (optional)
    ):
        self.heatmap_filter = heatmap_filter
        self.anomaly_detector = anomaly_detector
        self.yolo_detector = yolo_detector
    
    def process_alert(
        self,
        camera_id: str,
        alert_id: str,
        images: List[np.ndarray],
        motion_mask: np.ndarray,
        motion_boxes: List[Tuple[int, int, int, int]],
    ) -> Dict[str, object]:
        """
        Process an alert through the full pipeline.
        
        Args:
            camera_id: Camera identifier
            alert_id: Alert identifier
            images: [I1, I2, I3] three consecutive frames
            motion_mask: Binary motion mask for the alert
            motion_boxes: List of detected motion bounding boxes
        
        Returns:
            Dict with pipeline results:
            - stage_3_result: HeatMap filter output
            - stage_4_result: Memory-based anomaly detection
            - stage_5_result: YOLO detections (if applicable)
            - final_decision: "ESCALATE" or "DROP"
            - explanation: Human-readable explanation
        """
        result = {
            "alert_id": alert_id,
            "camera_id": camera_id,
            "timestamp": datetime.now().isoformat(),
            "stage_3_result": None,
            "stage_4_result": None,
            "stage_5_result": None,
            "final_decision": "DROP",
            "explanation": "",
        }
        
        # ========== STAGE 3: Heat Map Analysis ==========
        try:
            stage3_result = self.heatmap_filter.process_event(
                motion_mask=motion_mask,
                original_image=images[1],  # Use I2
                save_overlay=False,
            )
            result["stage_3_result"] = stage3_result
            
            heat_zone = stage3_result.get("heat_zone", "hot")
            next_stage_hint = stage3_result.get("next_stage", 4)
            anomaly_score = stage3_result.get("anomaly_score", 0.0)
            
        except Exception as e:
            result["stage_3_result"] = {"error": str(e)}
            result["explanation"] += f"Stage 3 error: {e}. "
            return result
        
        # ========== STAGE 4: Memory-based Anomaly Detection ==========
        try:
            stage4_result = self.anomaly_detector.detect(
                camera_id=camera_id,
                images=images,
                motion_boxes=motion_boxes,
                top_k=10,
            )
            result["stage_4_result"] = stage4_result
            
            event_sim = stage4_result.get("event_sim", 0.0)
            event_support = stage4_result.get("event_support", 0)
            stage4_decision = stage4_result.get("decision", "PASS")
            
        except Exception as e:
            result["stage_4_result"] = {"error": str(e)}
            result["explanation"] += f"Stage 4 error: {e}. "
            # If Stage 4 fails, fall back to Stage 3 decision
            if next_stage_hint == 5:
                result["final_decision"] = "ESCALATE"
            return result
        
        # ========== Decision Logic ==========
        # Stage 4 FILTER → ESCALATE
        if stage4_decision == "FILTER":
            result["final_decision"] = "ESCALATE"
            result["explanation"] = (
                f"Stage 4: High similarity ({event_sim:.3f}) with {event_support} "
                f"past events (anomalous). "
            )
            
            # ========== STAGE 5: YOLO (if available) ==========
            if self.yolo_detector and not motion_boxes:
                try:
                    yolo_result = self.yolo_detector.detect(
                        images[1],  # Use I2
                        confidence=0.6,
                    )
                    result["stage_5_result"] = yolo_result
                    result["explanation"] += f"YOLO detected: {len(yolo_result.get('detections', []))} objects. "
                except Exception as e:
                    result["stage_5_result"] = {"error": str(e)}
                    result["explanation"] += f"Stage 5 error: {e}. "
        
        # Stage 4 DROP → DROP
        elif stage4_decision == "DROP":
            result["final_decision"] = "DROP"
            result["explanation"] = (
                f"Stage 4: Low similarity ({event_sim:.3f}); "
                f"only {event_support} similar neighbors. Novel or false alarm."
            )
        
        # Stage 4 PASS → Use Stage 3 hint
        else:  # PASS
            if next_stage_hint == 5:
                result["final_decision"] = "ESCALATE"
                result["explanation"] = (
                    f"Stage 3 hint: heat_zone='{heat_zone}' → Stage 5. "
                    f"Stage 4: Moderate match ({event_sim:.3f}). "
                )
            else:
                result["final_decision"] = "DROP"
                result["explanation"] = (
                    f"Stage 3 hint: heat_zone='{heat_zone}' → Stage 4. "
                    f"Stage 4: Uncertain match ({event_sim:.3f}). Holding as normal."
                )
        
        return result


# Integration example and utilities
def create_pipeline(
    heatmap_filter,
    anomaly_detector,
    yolo_detector=None,
) -> AlertProcessingPipeline:
    """Create a configured alert processing pipeline."""
    return AlertProcessingPipeline(
        heatmap_filter=heatmap_filter,
        anomaly_detector=anomaly_detector,
        yolo_detector=yolo_detector,
    )
