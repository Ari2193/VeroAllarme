"""
Filter module that uses the Stage 3 Heat Map.

Part 1: Filter that consumes motion masks, ensures the heat map exists,
updates history, calculates anomaly scores, and persists state.
Part 2 (existing): The heat map implementation in `heatmap.py`.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

from .heatmap import HeatMapAnalyzer


class FilterHeatMap:
    """
    Uses a persistent `HeatMapAnalyzer` to filter motion events based on
    historical spatial patterns.
    """

    def __init__(
        self,
        camera_id: str = "default",
        storage_path: str = "data/heatmaps",
        frame_shape: Tuple[int, int] = (480, 640),
        history_days: int = 30,
        decay_factor: float = 0.95,
        anomaly_threshold: float = 0.6,
        normalization: str = "max"
    ) -> None:
        self.camera_id = camera_id
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.frame_shape = frame_shape
        self.history_days = history_days
        self.decay_factor = decay_factor
        self.anomaly_threshold = anomaly_threshold
        self.normalization = normalization

        # Initialize analyzer (auto-loads if pickle exists)
        self.analyzer = HeatMapAnalyzer(
            storage_path=str(self.storage_path),
            history_days=self.history_days,
            frame_shape=self.frame_shape,
            decay_factor=self.decay_factor,
        )
        # Try to load camera-specific heat map if present
        try:
            self.analyzer._load_heatmap(camera_id=self.camera_id)  # noqa: SLF001 (internal ok here)
        except Exception:
            pass

    def ensure_initialized(self) -> None:
        """Ensure internal analyzer is ready (creates empty map if needed)."""
        # HeatMapAnalyzer already initializes an empty map if none found.
        # This method left intentionally for symmetry/explicit call sites.
        expected_shape = (*self.frame_shape, self.analyzer.n_time_bins)
        if self.analyzer.heat_map.ndim != 3 or self.analyzer.heat_map.shape != expected_shape:
            self.analyzer.heat_map = np.zeros(expected_shape, dtype=np.float32)

    def process_event(
        self,
        motion_mask: np.ndarray,
        original_image: Optional[np.ndarray] = None,
        save_overlay: bool = False,
        overlay_path: Optional[Path] = None,
    ) -> Dict[str, object]:
        """
        Process a single motion event:
        - Ensure heat map exists (initialize if needed)
        - Add motion to history (with decay)
        - Calculate anomaly score
        - Persist updated heat map
        - Optionally generate an overlay image

        Returns:
            Dict with keys: anomaly_score, flagged, heat_zone, next_stage,
            stats, overlay_path(optional)
        """
        self.ensure_initialized()

        # Add event to history
        now = datetime.now()
        self.analyzer.add_motion_event(motion_mask, timestamp=now)

        # Calculate anomaly scores
        scores = self.analyzer.calculate_anomaly_score(motion_mask, timestamp=now)
        flagged = scores["anomaly_score"] > self.anomaly_threshold

        # Decide next stage: cold (unusual) -> Stage 5 (YOLO), hot (expected) -> Stage 4
        heat_zone = "cold" if flagged else "hot"
        next_stage = 5 if heat_zone == "cold" else 4

        # Persist
        self.analyzer.save_heatmap(camera_id=self.camera_id)

        result: Dict[str, object] = {
            "anomaly_score": scores["anomaly_score"],
            "flagged": bool(flagged),
            "heat_zone": heat_zone,
            "next_stage": next_stage,
            "stats": self.analyzer.get_statistics(),
        }

        # Optional overlay generation
        if save_overlay and original_image is not None:
            try:
                overlay_img = self.analyzer.generate_heatmap_overlay(original_image, alpha=0.4, timestamp=now)
                if overlay_path is None:
                    overlay_dir = Path("demo_output")
                    overlay_dir.mkdir(exist_ok=True)
                    overlay_path = overlay_dir / f"heatmap_overlay_{self.camera_id}.jpg"
                if cv2 is None:
                    result["overlay_error"] = "opencv-not-installed"
                else:
                    cv2.imwrite(str(overlay_path), overlay_img)
                    result["overlay_path"] = str(overlay_path)
            except RuntimeError as e:
                result["overlay_error"] = str(e)

        return result

    def process_batch(self, motion_masks: List[np.ndarray]) -> np.ndarray:
        """Add a batch of masks and return the normalized heat map."""
        self.ensure_initialized()
        _ = self.analyzer.build_from_batch(motion_masks, normalization=self.normalization)
        self.analyzer.save_heatmap(camera_id=self.camera_id)
        return self.analyzer.get_probability_heatmap(method=self.normalization)

    def get_heatmap(self) -> np.ndarray:
        """Return the current normalized heat map."""
        return self.analyzer.get_probability_heatmap(method=self.normalization)

    def get_overlay(self, original_image: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Return a heat map overlay for a given image."""
        return self.analyzer.generate_heatmap_overlay(original_image, alpha=alpha)
