"""
Tests for Stage 4: Memory-first anomaly detection.
"""

import pytest
import numpy as np
from datetime import datetime
from pathlib import Path

# Import Stage 4 components
# Note: These imports will work once FAISS and transformers are available
# For now, they serve as documentation of the intended API


def test_event_creation():
    """Test Event representation."""
    from backend.core.anomaly_detection import Event
    
    event_id = "evt_001"
    camera_id = "cam_1"
    timestamp = datetime.now()
    motion_boxes = [(10, 20, 100, 120), (150, 50, 250, 180)]
    embeddings = np.random.randn(2, 512).astype(np.float32)
    
    event = Event(
        event_id=event_id,
        camera_id=camera_id,
        timestamp=timestamp,
        motion_boxes=motion_boxes,
        embeddings=embeddings,
        label="true_event",
    )
    
    assert event.event_id == event_id
    assert event.camera_id == camera_id
    assert event.num_boxes == 2
    assert event.embeddings.shape == (2, 512)
    assert event.label == "true_event"


@pytest.mark.skipif(
    True,  # Skip until FAISS is available
    reason="Requires FAISS and transformers libraries"
)
def test_memory_index_add_retrieve(tmp_path):
    """Test MemoryIndex add and retrieve operations."""
    from backend.core.anomaly_detection import MemoryIndex, Event
    
    camera_id = "cam_test"
    index = MemoryIndex(
        camera_id=camera_id,
        embedding_dim=512,
        storage_path=str(tmp_path),
    )
    
    # Add events
    for i in range(5):
        event = Event(
            event_id=f"evt_{i:03d}",
            camera_id=camera_id,
            timestamp=datetime.now(),
            motion_boxes=[(10 * i, 20 * i, 100 + 10 * i, 120 + 20 * i)],
            embeddings=np.random.randn(1, 512).astype(np.float32),
            label="true_event",
        )
        index.add_event(event)
    
    # Query
    query_emb = np.random.randn(1, 512).astype(np.float32)
    sims, idxs = index.retrieve_neighbors(query_emb, k=3)
    
    assert sims.shape == (1, 3)
    assert idxs.shape == (1, 3)


@pytest.mark.skipif(
    True,
    reason="Requires FAISS and transformers libraries"
)
def test_anomaly_detector_decision_rule(tmp_path):
    """Test AnomalyDetector decision logic."""
    from backend.core.anomaly_detection import AnomalyDetector
    
    detector = AnomalyDetector(
        sim_strong=0.92,
        sim_weak=0.85,
        support_min=8,
        storage_path=str(tmp_path),
    )
    
    # Create dummy images and motion boxes
    images = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    ]
    motion_boxes = [(100, 100, 200, 200)]
    
    result = detector.detect("cam_1", images, motion_boxes)
    
    assert "event_sim" in result
    assert "event_support" in result
    assert "decision" in result
    assert result["decision"] in ["FILTER", "DROP", "PASS"]


def test_anomaly_detector_empty_boxes(tmp_path):
    """Test handling of empty motion boxes."""
    from backend.core.anomaly_detection import AnomalyDetector
    
    detector = AnomalyDetector(storage_path=str(tmp_path))
    
    images = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    ]
    
    result = detector.detect("cam_1", images, [])
    
    assert result["decision"] == "DROP"
    assert result["event_support"] == 0
    assert "No motion boxes" in result["reasoning"]


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
