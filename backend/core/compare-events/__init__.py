"""
Stage 4: Anomaly Detection via Event Comparison (Memory-First Similarity Matching)

This module provides memory-based filtering using embeddings and FAISS indexing
to compare motion events against historical patterns.

Main classes:
- Event: Represents a motion event with embeddings
- MemoryIndex: Per-camera FAISS index for fast similarity search
- AnomalyDetector: Main detector using similarity and support scoring

Usage:
    from backend.core.compare_events import AnomalyDetector
    
    detector = AnomalyDetector()
    result = detector.detect(camera_id, images, motion_boxes)
    print(result["decision"])  # "FILTER", "DROP", or "PASS"
"""

try:
    from .anomaly_detection import Event, MemoryIndex, AnomalyDetector
except ImportError:
    # Handle case where relative import fails (e.g., during test discovery)
    try:
        from anomaly_detection import Event, MemoryIndex, AnomalyDetector
    except ImportError:
        pass

__all__ = ["Event", "MemoryIndex", "AnomalyDetector"]
