"""
Unit tests for motion detection module
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import List

from backend.core.motion_detection import (
    MotionDetector,
    MotionRegion,
    MotionDetectionResult,
    detect_motion_from_paths
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test images"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def static_images(temp_dir) -> List[str]:
    """Create three identical images (no motion)"""
    image_paths = []
    
    # Create a simple test image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)
    
    for i in range(3):
        path = temp_dir / f"static_{i}.jpg"
        cv2.imwrite(str(path), img)
        image_paths.append(str(path))
    
    return image_paths


@pytest.fixture
def motion_images(temp_dir) -> List[str]:
    """Create three images with motion"""
    image_paths = []
    
    for i in range(3):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Moving rectangle
        x_pos = 100 + (i * 50)
        cv2.rectangle(img, (x_pos, 100), (x_pos + 100, 200), (255, 0, 0), -1)
        
        path = temp_dir / f"motion_{i}.jpg"
        cv2.imwrite(str(path), img)
        image_paths.append(str(path))
    
    return image_paths


class TestMotionRegion:
    """Test MotionRegion dataclass"""
    
    def test_motion_region_creation(self):
        region = MotionRegion(
            x=100,
            y=150,
            width=50,
            height=75,
            area=3750,
            centroid=(125, 187)
        )
        
        assert region.x == 100
        assert region.y == 150
        assert region.width == 50
        assert region.height == 75
        assert region.area == 3750
        assert region.centroid == (125, 187)
    
    def test_to_dict(self):
        region = MotionRegion(
            x=100, y=150, width=50, height=75,
            area=3750, centroid=(125, 187)
        )
        
        result = region.to_dict()
        
        assert result["x"] == 100
        assert result["y"] == 150
        assert result["width"] == 50
        assert result["height"] == 75
        assert result["area"] == 3750
        assert result["centroid"]["x"] == 125
        assert result["centroid"]["y"] == 187
    
    def test_to_bbox(self):
        region = MotionRegion(
            x=100, y=150, width=50, height=75,
            area=3750, centroid=(125, 187)
        )
        
        bbox = region.to_bbox()
        
        assert bbox == (100, 150, 150, 225)


class TestMotionDetectionResult:
    """Test MotionDetectionResult dataclass"""
    
    def test_result_to_dict(self):
        region = MotionRegion(
            x=100, y=150, width=50, height=75,
            area=3750, centroid=(125, 187)
        )
        
        result = MotionDetectionResult(
            motion_detected=True,
            motion_regions=[region],
            motion_mask=np.zeros((480, 640)),
            confidence=0.85,
            total_motion_area=5000,
            frame_dimensions=(480, 640)
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["motion_detected"] is True
        assert result_dict["confidence"] == 0.85
        assert result_dict["total_motion_area"] == 5000
        assert result_dict["frame_dimensions"]["height"] == 480
        assert result_dict["frame_dimensions"]["width"] == 640
        assert result_dict["num_regions"] == 1


class TestMotionDetector:
    """Test MotionDetector class"""
    
    def test_initialization(self):
        detector = MotionDetector(
            threshold=30,
            min_area=600,
            blur_kernel=(15, 15),
            morph_kernel_size=7
        )
        
        assert detector.threshold == 30
        assert detector.min_area == 600
        assert detector.blur_kernel == (15, 15)
        assert detector.morph_kernel_size == 7
    
    def test_load_images_success(self, static_images):
        detector = MotionDetector()
        images = detector.load_images(static_images)
        
        assert len(images) == 3
        assert all(isinstance(img, np.ndarray) for img in images)
        assert all(img.shape == images[0].shape for img in images)
    
    def test_load_images_insufficient(self):
        detector = MotionDetector()
        
        with pytest.raises(ValueError, match="At least 2 images"):
            detector.load_images(["single_image.jpg"])
    
    def test_load_images_invalid_path(self):
        detector = MotionDetector()
        
        with pytest.raises(ValueError, match="Failed to load"):
            detector.load_images(["nonexistent1.jpg", "nonexistent2.jpg"])
    
    def test_preprocess_frame(self):
        detector = MotionDetector()
        
        # Create test image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        processed = detector.preprocess_frame(img)
        
        assert processed.ndim == 2  # Grayscale
        assert processed.shape == (480, 640)
        assert processed.dtype == np.uint8
    
    def test_compute_frame_difference(self):
        detector = MotionDetector(threshold=25)
        
        # Create two different frames
        frame1 = np.ones((480, 640), dtype=np.uint8) * 100
        frame2 = np.ones((480, 640), dtype=np.uint8) * 100
        frame2[100:200, 100:200] = 200  # Add difference
        
        diff = detector.compute_frame_difference(frame1, frame2)
        
        assert diff.shape == (480, 640)
        assert np.any(diff > 0)  # Should detect difference
    
    def test_find_motion_regions(self):
        detector = MotionDetector(min_area=100)
        
        # Create motion mask with clear regions
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(mask, (100, 100), (200, 200), 255, -1)
        cv2.rectangle(mask, (300, 300), (350, 350), 255, -1)
        
        regions = detector.find_motion_regions(mask)
        
        assert len(regions) == 2
        assert all(isinstance(r, MotionRegion) for r in regions)
        assert regions[0].area >= regions[1].area  # Sorted by area
    
    def test_detect_motion_no_motion(self, static_images):
        detector = MotionDetector(threshold=25, min_area=500)
        result = detector.detect_motion(static_images)
        
        assert isinstance(result, MotionDetectionResult)
        assert result.motion_detected is False
        assert len(result.motion_regions) == 0
        assert result.confidence >= 0.0
    
    def test_detect_motion_with_motion(self, motion_images):
        detector = MotionDetector(threshold=25, min_area=500)
        result = detector.detect_motion(motion_images)
        
        assert isinstance(result, MotionDetectionResult)
        assert result.motion_detected is True
        assert len(result.motion_regions) > 0
        assert result.total_motion_area > 0
        assert result.confidence > 0.0
    
    def test_visualize_motion(self, motion_images, temp_dir):
        detector = MotionDetector(threshold=25, min_area=500)
        result = detector.detect_motion(motion_images)
        
        output_path = temp_dir / "visualization.jpg"
        
        vis = detector.visualize_motion(
            motion_images[1],
            result,
            output_path=str(output_path),
            show_mask=True,
            show_boxes=True,
            show_centroids=True
        )
        
        assert isinstance(vis, np.ndarray)
        assert vis.shape[2] == 3  # Color image
        assert Path(output_path).exists()


class TestConvenienceFunction:
    """Test convenience function"""
    
    def test_detect_motion_from_paths(self, motion_images):
        result = detect_motion_from_paths(
            motion_images,
            threshold=25,
            min_area=500
        )
        
        assert isinstance(result, MotionDetectionResult)
        assert result.motion_detected is True


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_different_image_sizes(self, temp_dir):
        # Create images with different sizes
        img1 = np.ones((480, 640, 3), dtype=np.uint8) * 128
        img2 = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        
        path1 = temp_dir / "img1.jpg"
        path2 = temp_dir / "img2.jpg"
        
        cv2.imwrite(str(path1), img1)
        cv2.imwrite(str(path2), img2)
        
        detector = MotionDetector()
        
        with pytest.raises(ValueError, match="different dimensions"):
            detector.load_images([str(path1), str(path2)])
    
    def test_very_small_motion(self, temp_dir):
        # Create images with very small motion
        detector = MotionDetector(threshold=25, min_area=5000)
        
        images = []
        for i in range(2):
            img = np.ones((480, 640, 3), dtype=np.uint8) * 128
            cv2.rectangle(img, (100 + i, 100), (110 + i, 110), (255, 0, 0), -1)
            
            path = temp_dir / f"small_motion_{i}.jpg"
            cv2.imwrite(str(path), img)
            images.append(str(path))
        
        result = detector.detect_motion(images)
        
        # Should not detect due to min_area threshold
        assert result.motion_detected is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
