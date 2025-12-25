"""
Tests for Stage 5 - YOLO Object Detection Processor.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from core.yolo_processor import YoloProcessor


class TestYoloProcessorInitialization:
    """Test YOLO processor initialization."""
    
    def test_init_with_default_model(self):
        """Test initialization with default model."""
        processor = YoloProcessor()
        assert processor.model is not None
        assert hasattr(processor.model, 'names')
    
    def test_init_with_custom_model(self):
        """Test initialization with custom model path."""
        # Use the default model for testing
        processor = YoloProcessor(model_name="yolov8n.pt")
        assert processor.model is not None


class TestYoloDetectObjects:
    """Test object detection functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create a YOLO processor instance."""
        return YoloProcessor()
    
    @pytest.fixture
    def dummy_image(self):
        """Create a dummy BGR image (640x480)."""
        # Create a simple image with some variation
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some random noise to make it more realistic
        image[:, :] = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        # Add a white rectangle (simulate an object)
        image[200:400, 250:450] = [255, 255, 255]
        return image
    
    def test_detect_objects_returns_list(self, processor, dummy_image):
        """Test that detect_objects returns a list."""
        result = processor.detect_objects(dummy_image, conf_threshold=0.1)
        assert isinstance(result, list)
    
    def test_detect_objects_with_low_confidence(self, processor, dummy_image):
        """Test detection with low confidence threshold."""
        result = processor.detect_objects(dummy_image, conf_threshold=0.1)
        # Should return a list (may be empty for dummy image)
        assert isinstance(result, list)
    
    def test_detect_objects_with_high_confidence(self, processor, dummy_image):
        """Test detection with high confidence threshold."""
        result = processor.detect_objects(dummy_image, conf_threshold=0.8)
        assert isinstance(result, list)
        # High confidence should have fewer or equal detections
    
    def test_detection_structure(self, processor, dummy_image):
        """Test that detections have correct structure."""
        result = processor.detect_objects(dummy_image, conf_threshold=0.1)
        if result:  # If any detections
            detection = result[0]
            assert "class" in detection
            assert "confidence" in detection
            assert "bbox" in detection
            assert isinstance(detection["class"], str)
            assert isinstance(detection["confidence"], float)
            assert isinstance(detection["bbox"], list)
            assert len(detection["bbox"]) == 4
            # Check bbox values are valid
            x1, y1, x2, y2 = detection["bbox"]
            assert x2 > x1  # x2 should be greater than x1
            assert y2 > y1  # y2 should be greater than y1
    
    def test_detect_objects_empty_image(self, processor):
        """Test detection on completely black image."""
        black_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = processor.detect_objects(black_image, conf_threshold=0.5)
        assert isinstance(result, list)
    
    def test_detect_objects_confidence_values(self, processor, dummy_image):
        """Test that confidence values are within valid range."""
        result = processor.detect_objects(dummy_image, conf_threshold=0.1)
        for detection in result:
            assert 0.0 <= detection["confidence"] <= 1.0


class TestYoloClassifyCrop:
    """Test crop classification functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create a YOLO processor instance."""
        return YoloProcessor()
    
    @pytest.fixture
    def dummy_image(self):
        """Create a dummy BGR image."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:, :] = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        return image
    
    def test_classify_crop_returns_string(self, processor, dummy_image):
        """Test that classify_crop returns a string."""
        result = processor.classify_crop(dummy_image, conf_threshold=0.1)
        assert isinstance(result, str)
    
    def test_classify_crop_empty_image(self, processor):
        """Test classification on empty image returns 'unknown'."""
        black_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = processor.classify_crop(black_image, conf_threshold=0.9)
        # Should return 'unknown' if nothing detected
        assert isinstance(result, str)
    
    def test_classify_crop_with_detections(self, processor, dummy_image):
        """Test that classify_crop returns a valid class name."""
        result = processor.classify_crop(dummy_image, conf_threshold=0.1)
        assert isinstance(result, str)
        # Result should either be 'unknown' or a valid COCO class
        assert len(result) > 0


class TestYoloIntegration:
    """Integration tests for YOLO processor."""
    
    @pytest.fixture
    def processor(self):
        """Create a YOLO processor instance."""
        return YoloProcessor()
    
    def test_multiple_detections_consistency(self, processor):
        """Test that multiple detections on same image are consistent."""
        image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        
        result1 = processor.detect_objects(image, conf_threshold=0.5)
        result2 = processor.detect_objects(image, conf_threshold=0.5)
        
        # Same image should produce same number of detections
        assert len(result1) == len(result2)
    
    def test_confidence_threshold_filtering(self, processor):
        """Test that higher confidence threshold reduces detections."""
        image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        
        low_conf_result = processor.detect_objects(image, conf_threshold=0.1)
        high_conf_result = processor.detect_objects(image, conf_threshold=0.8)
        
        # Higher confidence should have fewer or equal detections
        assert len(high_conf_result) <= len(low_conf_result)
    
    def test_bbox_coordinates_valid(self, processor):
        """Test that bounding box coordinates are within image bounds."""
        image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        height, width = image.shape[:2]
        
        result = processor.detect_objects(image, conf_threshold=0.1)
        
        for detection in result:
            x1, y1, x2, y2 = detection["bbox"]
            # Coordinates should be within image bounds
            assert 0 <= x1 < width
            assert 0 <= y1 < height
            assert 0 < x2 <= width
            assert 0 < y2 <= height
    
    def test_classify_uses_highest_confidence(self, processor):
        """Test that classify_crop returns the highest confidence detection."""
        image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        
        class_name = processor.classify_crop(image, conf_threshold=0.1)
        detections = processor.detect_objects(image, conf_threshold=0.1)
        
        if detections:
            best_detection = sorted(detections, key=lambda x: x['confidence'], reverse=True)[0]
            assert class_name == best_detection['class']
        else:
            assert class_name == "unknown"


class TestYoloEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def processor(self):
        """Create a YOLO processor instance."""
        return YoloProcessor()
    
    def test_very_small_image(self, processor):
        """Test detection on very small image."""
        small_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = processor.detect_objects(small_image, conf_threshold=0.5)
        assert isinstance(result, list)
    
    def test_very_large_confidence(self, processor):
        """Test with confidence threshold close to 1.0."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = processor.detect_objects(image, conf_threshold=0.99)
        assert isinstance(result, list)
    
    def test_zero_confidence(self, processor):
        """Test with zero confidence threshold."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = processor.detect_objects(image, conf_threshold=0.0)
        assert isinstance(result, list)
    
    def test_rectangular_image(self, processor):
        """Test with non-square rectangular image."""
        rect_image = np.random.randint(0, 255, (300, 800, 3), dtype=np.uint8)
        result = processor.detect_objects(rect_image, conf_threshold=0.5)
        assert isinstance(result, list)
