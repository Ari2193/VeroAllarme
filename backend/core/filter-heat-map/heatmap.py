"""
Stage 3: Heat Map Analysis (Historical Patterns)

This module builds spatial probability maps of where significant motion typically occurs.
It maintains a rolling history of motion events and flags unusual patterns.
"""

import numpy as np
try:
    import cv2  # type: ignore
except Exception:  # OpenCV optional
    cv2 = None  # type: ignore
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, Tuple, Optional, List
import pickle


class HeatMapAnalyzer:
    """
    Analyzes motion patterns over time to build historical heat maps
    and detect anomalous motion in unusual locations.
    """
    
    def __init__(
        self,
        storage_path: str = "data/heatmaps",
        history_days: int = 30,
        frame_shape: Tuple[int, int] = (480, 640),
        decay_factor: float = 0.95,
        time_binning: str = "day_night",
        n_time_bins: Optional[int] = None,
        day_hours: Tuple[int, int] = (7, 19),
    ):
        """
        Initialize the heat map analyzer.
        
        Args:
            storage_path: Directory to store heat map data
            history_days: Number of days to maintain in rolling history (7-30)
            frame_shape: Expected shape of motion masks (height, width)
            decay_factor: Exponential decay for older events (0-1)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.history_days = history_days
        self.frame_shape = frame_shape
        self.decay_factor = decay_factor
        self.time_binning = time_binning  # 'day_night' | 'hour'
        # Determine number of bins
        if n_time_bins is not None:
            self.n_time_bins = int(n_time_bins)
        else:
            self.n_time_bins = 2 if self.time_binning == "day_night" else 24
        self.day_hours = day_hours  # used for day_night
        
        # Initialize heat map accumulator (H, W, T)
        self.heat_map = np.zeros((*frame_shape, self.n_time_bins), dtype=np.float32)
        
        # Track event count (total) and per-bin counts
        self.event_count = 0
        self.per_bin_event_count = [0 for _ in range(self.n_time_bins)]
        
        # Metadata for tracking
        self.last_update = datetime.now()
        self.events_history = []  # List of (timestamp, motion_data)
        
        # Load existing heat map if available
        self._load_heatmap()
    
    def add_motion_event(self, motion_mask: np.ndarray, timestamp: Optional[datetime] = None) -> None:
        """
        Add a new motion event to the historical heat map.
        
        Args:
            motion_mask: Binary motion mask (0 or 255) from Stage 1/2
            timestamp: When the event occurred (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Ensure motion mask is binary and correct shape
        if motion_mask.shape != self.frame_shape:
            if cv2 is None:
                raise ValueError(
                    "OpenCV not available: provide motion_mask in frame_shape or install opencv-python"
                )
            motion_mask = cv2.resize(motion_mask, (self.frame_shape[1], self.frame_shape[0]))
        
        # Convert to binary (0 or 1)
        binary_mask = (motion_mask > 127).astype(np.float32)
        
        # Apply temporal decay on every event to favor recent history
        self._apply_decay()
        # Route to the correct time bin
        bin_idx = self._get_time_bin(timestamp)
        # Add current event to that bin
        self.heat_map[:, :, bin_idx] += binary_mask
        self.event_count += 1
        self.per_bin_event_count[bin_idx] += 1
        
        # Store event metadata
        self.events_history.append({
            'timestamp': timestamp,
            'motion_area': np.sum(binary_mask)
        })
        
        # Clean old events
        self._cleanup_old_events()
        
        # (Decay already applied per event)
        
        self.last_update = timestamp
    
    def get_probability_heatmap(self, method: str = "max", *, for_timestamp: Optional[datetime] = None, time_bin: Optional[int] = None, aggregate: str = "bin") -> np.ndarray:
        """
        Get a normalized probability heat map (0-1 scale).
        
        Returns:
            2D Normalized heat map where each pixel represents
            the probability of motion occurrence at that location.
        Args:
            method: 'max' (default) or 'events'
            for_timestamp: if provided, selects the bin for that time
            time_bin: explicit time bin index to use
            aggregate: 'bin' (default), 'max_all', or 'mean_all'
        """
        if self.event_count == 0:
            return np.zeros(self.frame_shape, dtype=np.float32)

        # Decide which slice to use or aggregate
        if aggregate in ("max_all", "mean_all"):
            slice_map = self.heat_map
            if method == "events":
                denom = float(max(1, self.event_count))
                norm = np.clip(slice_map / denom, 0.0, 1.0)
            else:
                max_val = float(np.max(slice_map))
                norm = slice_map / max_val if max_val > 0 else slice_map
            if aggregate == "max_all":
                return np.max(norm, axis=2)
            else:
                return np.mean(norm, axis=2)

        # Use a specific bin
        if time_bin is None:
            bin_idx = self._get_time_bin(for_timestamp or self.last_update)
        else:
            bin_idx = int(time_bin) % self.n_time_bins

        slice_map = self.heat_map[:, :, bin_idx]
        if method == "events":
            denom = float(max(1, self.per_bin_event_count[bin_idx]))
            norm = np.clip(slice_map / denom, 0.0, 1.0)
        else:
            max_val = float(np.max(slice_map))
            norm = slice_map / max_val if max_val > 0 else slice_map
        return norm
    
    def calculate_anomaly_score(self, motion_mask: np.ndarray, *, timestamp: Optional[datetime] = None) -> Dict[str, float]:
        """
        Compare current motion against historical baseline and calculate anomaly score.
        
        Args:
            motion_mask: Current motion mask to analyze
            timestamp: time used to select the appropriate time bin
        
        Returns:
            Dictionary with:
                - anomaly_score: 0-1, higher means more unusual
                - motion_in_quiet_zones: percentage of motion in low-probability areas
                - motion_in_active_zones: percentage of motion in high-probability areas
                - overall_motion_area: total pixels with motion
        """
        # Ensure correct shape
        if motion_mask.shape != self.frame_shape:
            motion_mask = cv2.resize(motion_mask, (self.frame_shape[1], self.frame_shape[0]))
        
        # Convert to binary
        binary_mask = (motion_mask > 127).astype(np.float32)
        
        # Get probability heat map for this time bin
        prob_map = self.get_probability_heatmap(for_timestamp=timestamp, aggregate="bin")
        
        # Calculate motion pixels
        motion_pixels = binary_mask > 0
        total_motion_area = np.sum(motion_pixels)
        
        if total_motion_area == 0:
            return {
                'anomaly_score': 0.0,
                'motion_in_quiet_zones': 0.0,
                'motion_in_active_zones': 0.0,
                'overall_motion_area': 0
            }
        
        # Define quiet zones (low historical activity) and active zones
        quiet_threshold = 0.3  # Pixels with <30% probability
        active_threshold = 0.7  # Pixels with >70% probability
        
        quiet_zones = prob_map < quiet_threshold
        active_zones = prob_map > active_threshold
        
        # Calculate overlap
        motion_in_quiet = np.sum(motion_pixels & quiet_zones)
        motion_in_active = np.sum(motion_pixels & active_zones)
        
        # Calculate percentages
        quiet_percent = (motion_in_quiet / total_motion_area) * 100
        active_percent = (motion_in_active / total_motion_area) * 100
        
        # Anomaly score: higher when motion is in quiet zones
        # Lower when motion is in expected (active) zones
        if self.event_count < 10:
            # Not enough history, use moderate score
            anomaly_score = 0.5
        else:
            # Weight towards quiet zone activity
            anomaly_score = (quiet_percent / 100) * 0.8 + (1 - active_percent / 100) * 0.2
            
            # Clamp to 0-1
            anomaly_score = np.clip(anomaly_score, 0.0, 1.0)
        
        return {
            'anomaly_score': float(anomaly_score),
            'motion_in_quiet_zones': float(quiet_percent),
            'motion_in_active_zones': float(active_percent),
            'overall_motion_area': int(total_motion_area)
        }
    
    def generate_heatmap_overlay(
        self,
        original_image: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.4,
        *,
        timestamp: Optional[datetime] = None,
        time_bin: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a visual heat map overlay on the original image.
        
        Args:
            original_image: Original camera frame
            colormap: OpenCV colormap (COLORMAP_JET, COLORMAP_HOT, etc.)
            alpha: Transparency of overlay (0-1)
        
        Returns:
            Image with heat map overlay
        """
        # Get normalized heat map for selected bin
        if time_bin is not None or timestamp is not None:
            prob_map = self.get_probability_heatmap(for_timestamp=timestamp, time_bin=time_bin, aggregate="bin")
        else:
            prob_map = self.get_probability_heatmap(aggregate="bin")
        
        # Convert to uint8 for colormap
        heatmap_uint8 = (prob_map * 255).astype(np.uint8)
        
        if cv2 is None:
            raise RuntimeError("OpenCV not available: overlay generation requires opencv-python")
        # Apply colormap (red = high probability)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        
        # Resize to match original image if needed
        if colored_heatmap.shape[:2] != original_image.shape[:2]:
            colored_heatmap = cv2.resize(
                colored_heatmap,
                (original_image.shape[1], original_image.shape[0])
            )
        
        # Ensure original image is BGR
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        
        # Blend with original image
        overlay = cv2.addWeighted(original_image, 1 - alpha, colored_heatmap, alpha, 0)
        
        return overlay

    def build_from_batch(self, motion_masks: List[np.ndarray], normalization: str = "max") -> np.ndarray:
        """
        Build a heat map from a batch of motion masks.
        
        Args:
            motion_masks: List of binary motion masks (0 or 255)
            normalization: 'max' or 'events' normalization method
        
        Returns:
            Normalized heat map (0-1)
        """
        for mask in motion_masks:
            self.add_motion_event(mask)
        return self.get_probability_heatmap(method=normalization)

    def reset(self) -> None:
        """Reset the heat map and history."""
        self.heat_map = np.zeros(self.frame_shape, dtype=np.float32)
        self.event_count = 0
        self.events_history = []
        self.last_update = datetime.now()
    
    def flag_unusual_motion(self, motion_mask: np.ndarray, threshold: float = 0.6) -> bool:
        """
        Flag motion in unusual locations (low probability zones).
        
        Args:
            motion_mask: Current motion mask
            threshold: Anomaly score threshold (0-1) to flag as unusual
        
        Returns:
            True if motion is unusual, False otherwise
        """
        scores = self.calculate_anomaly_score(motion_mask)
        return scores['anomaly_score'] > threshold
    
    def _cleanup_old_events(self) -> None:
        """Remove events older than history_days."""
        cutoff_date = datetime.now() - timedelta(days=self.history_days)
        
        # Filter out old events
        self.events_history = [
            event for event in self.events_history
            if event['timestamp'] > cutoff_date
        ]
    
    def _apply_decay(self) -> None:
        """Apply exponential decay to heat map for temporal relevance."""
        self.heat_map *= self.decay_factor
    
    def save_heatmap(self, camera_id: str = "default") -> None:
        """
        Save current heat map and metadata to disk.
        
        Args:
            camera_id: Identifier for the camera (for multi-camera setups)
        """
        save_path = self.storage_path / f"heatmap_{camera_id}.pkl"
        
        data = {
            'heat_map': self.heat_map,
            'event_count': self.event_count,
            'per_bin_event_count': self.per_bin_event_count,
            'last_update': self.last_update,
            'events_history': self.events_history,
            'frame_shape': self.frame_shape,
            'history_days': self.history_days,
            'time_binning': self.time_binning,
            'n_time_bins': self.n_time_bins,
            'day_hours': self.day_hours,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
    
    def _load_heatmap(self, camera_id: str = "default") -> bool:
        """
        Load existing heat map from disk if available.
        
        Args:
            camera_id: Identifier for the camera
        
        Returns:
            True if loaded successfully, False otherwise
        """
        load_path = self.storage_path / f"heatmap_{camera_id}.pkl"
        
        if not load_path.exists():
            return False
        
        try:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
            
            loaded_map = data.get('heat_map')
            # Backward compatibility: upgrade 2D -> 3D with single bin
            if loaded_map.ndim == 2:
                loaded_map = np.expand_dims(loaded_map.astype(np.float32), axis=2)
            self.heat_map = loaded_map
            self.event_count = data['event_count']
            self.per_bin_event_count = data.get('per_bin_event_count', [self.event_count])
            self.last_update = data['last_update']
            self.events_history = data['events_history']
            self.time_binning = data.get('time_binning', self.time_binning)
            self.n_time_bins = int(data.get('n_time_bins', self.heat_map.shape[2]))
            self.day_hours = data.get('day_hours', self.day_hours)
            # Align shapes/config if mismatch
            if self.heat_map.shape[:2] != self.frame_shape:
                if cv2 is None:
                    raise RuntimeError(
                        "OpenCV not available: cannot resize loaded heat map to frame_shape"
                    )
                self.heat_map = np.stack([
                    cv2.resize(self.heat_map[:, :, i], (self.frame_shape[1], self.frame_shape[0]))
                    for i in range(self.heat_map.shape[2])
                ], axis=2).astype(np.float32)
            if self.heat_map.shape[2] != self.n_time_bins:
                # Pad or truncate bins to match n_time_bins
                current_T = self.heat_map.shape[2]
                if current_T < self.n_time_bins:
                    pad = np.zeros((*self.frame_shape, self.n_time_bins - current_T), dtype=np.float32)
                    self.heat_map = np.concatenate([self.heat_map, pad], axis=2)
                    self.per_bin_event_count += [0] * (self.n_time_bins - current_T)
                else:
                    self.heat_map = self.heat_map[:, :, : self.n_time_bins]
                    self.per_bin_event_count = self.per_bin_event_count[: self.n_time_bins]
            
            # Cleanup old events after loading
            self._cleanup_old_events()
            
            return True
        except Exception as e:
            print(f"Failed to load heat map: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the heat map.
        
        Returns:
            Dictionary with heat map statistics
        """
        # Use mean across bins for global stats
        prob_map = self.get_probability_heatmap(aggregate="mean_all")
        
        return {
            'total_events': self.event_count,
            'last_update': self.last_update.isoformat(),
            'history_span_days': self.history_days,
            'events_in_history': len(self.events_history),
            'max_probability': float(np.max(prob_map)),
            'mean_probability': float(np.mean(prob_map)),
            'active_pixels': int(np.sum(prob_map > 0.3)),
            'quiet_pixels': int(np.sum(prob_map < 0.1)),
            'n_time_bins': self.n_time_bins,
            'per_bin_event_count': list(self.per_bin_event_count),
        }

    def _get_time_bin(self, timestamp: datetime) -> int:
        """Map a timestamp to a time bin index."""
        if self.time_binning == "day_night":
            start_h, end_h = self.day_hours
            hour = timestamp.hour
            is_day = start_h <= hour < end_h
            # convention: bin 0 = day, bin 1 = night
            return 0 if is_day else 1
        # hour-of-day binning
        if self.n_time_bins == 24:
            return timestamp.hour
        # generic modulo mapping
        # map hour-of-day into n_time_bins
        return int((timestamp.hour / 24.0) * self.n_time_bins) % self.n_time_bins
