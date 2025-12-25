"""
Stage 4: Anomaly Detection (Memory-First Similarity Matching)

This module provides memory-based filtering using embeddings and FAISS indexing
to compare motion events against historical patterns.

Main classes:
- Event: Represents a motion event with embeddings
- MemoryIndex: Per-camera FAISS index for fast similarity search
- AnomalyDetector: Main detector using similarity and support scoring

Usage:
    from backend.core.stage4 import AnomalyDetector
    
    detector = AnomalyDetector()
    result = detector.detect(camera_id, images, motion_boxes)
    print(result["decision"])  # "FILTER", "DROP", or "PASS"
"""

from .anomaly_detection import Event, MemoryIndex, AnomalyDetector

__all__ = ["Event", "MemoryIndex", "AnomalyDetector"]
