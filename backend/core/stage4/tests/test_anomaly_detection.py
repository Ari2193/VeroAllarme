"""
Tests for Stage 4: Anomaly Detection

Test the memory-based filtering using embedding similarity.
"""

import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Handle imports from backend.core
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stage4 import Event, MemoryIndex, AnomalyDetector


def create_dummy_images(h=480, w=640, c=3):
    """Create dummy RGB images."""
    return [np.random.randint(0, 256, (h, w, c), dtype=np.uint8) for _ in range(3)]


def create_dummy_embeddings(n_boxes=1, embedding_dim=512):
    """Create dummy embeddings."""
    return np.random.randn(n_boxes, embedding_dim).astype(np.float32)


class TestEvent:
    """Test Event class."""
    
    def test_event_creation(self):
        """Test creating an Event object."""
        embeddings = create_dummy_embeddings(n_boxes=2)
        event = Event(
            event_id="event_001",
            camera_id="camera_1",
            timestamp=datetime.now(),
            motion_boxes=[(10, 20, 100, 150), (200, 300, 350, 400)],
            embeddings=embeddings,
            label="true_event"
        )
        
        assert event.event_id == "event_001"
        assert event.num_boxes == 2
        assert event.embeddings.shape == (2, 512)


class TestAnomalyDetector:
    """Test AnomalyDetector class."""
    
    def test_detector_creation(self):
        """Test creating an AnomalyDetector."""
        detector = AnomalyDetector(
            embedding_model="openai/clip-vit-base-patch32",
            embedding_dim=512,
            sim_strong=0.92,
            sim_weak=0.85,
            support_min=8,
        )
        assert detector is not None
        assert detector.sim_strong == 0.92
        assert detector.sim_weak == 0.85
    
    def test_detect_empty_boxes(self):
        """Test detecting with no motion boxes."""
        detector = AnomalyDetector()
        images = create_dummy_images()
        
        result = detector.detect(
            camera_id="camera_1",
            images=images,
            motion_boxes=[],
            top_k=10
        )
        
        assert result["decision"] == "DROP"
        assert result["event_support"] == 0
        assert "No motion boxes" in result["reasoning"]
    
    @pytest.mark.skipif(True, reason="Requires FAISS/transformers installation")
    def test_detect_with_empty_history(self):
        """Test detection when no historical data available."""
        detector = AnomalyDetector()
        images = create_dummy_images()
        motion_boxes = [(100, 100, 200, 200)]
        
        result = detector.detect(
            camera_id="camera_1",
            images=images,
            motion_boxes=motion_boxes,
            top_k=10
        )
        
        assert "decision" in result
        assert "event_sim" in result
        assert "event_support" in result
    
    @pytest.mark.skipif(True, reason="Requires FAISS/transformers installation")
    def test_add_event_to_memory(self):
        """Test adding an event to memory."""
        detector = AnomalyDetector()
        images = create_dummy_images()
        motion_boxes = [(100, 100, 200, 200)]
        
        detector.add_event_to_memory(
            camera_id="camera_1",
            event_id="event_001",
            images=images,
            motion_boxes=motion_boxes,
            label="true_event"
        )
        
        stats = detector.get_statistics("camera_1")
        assert "total_embeddings" in stats
        assert stats["unique_events"] >= 0
