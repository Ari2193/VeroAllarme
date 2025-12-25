"""
VeroAllarme - Tests for Relevance Mask Module

Comprehensive test suite for relevance mask functionality including:
- Model architecture tests
- Dataset loading tests
- Training pipeline tests
- Prediction API tests
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.relevance_mask import (
    RelevanceRegion,
    RelevanceMaskResult,
    ContrastiveEncoder,
    RelevanceSegmentationHead,
    RelevanceClassificationHead,
    RelevanceMaskModel,
    RelevanceMaskPredictor,
    predict_relevance_mask
)
from core.motion_detection import MotionDetector, MotionRegion, MotionDetectionResult


# ==================== Fixtures ====================

@pytest.fixture
def sample_image():
    """Create a sample BGR image"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_motion_result():
    """Create a sample motion detection result"""
    regions = [
        MotionRegion(x=100, y=100, width=50, height=60, area=3000, centroid=(125, 130)),
        MotionRegion(x=300, y=200, width=80, height=80, area=6400, centroid=(340, 240))
    ]
    
    binary_mask = np.zeros((480, 640), dtype=np.uint8)
    binary_mask[100:160, 100:150] = 1
    binary_mask[200:280, 300:380] = 1
    
    return MotionDetectionResult(
        frame_dimensions=(480, 640),
        motion_detected=True,
        binary_mask=binary_mask,
        motion_regions=regions,
        total_motion_area=9400,
        confidence=0.5
    )


@pytest.fixture
def temp_model_path():
    """Create a temporary model checkpoint"""
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        model = RelevanceMaskModel(mode="both")
        torch.save({"model_state_dict": model.state_dict()}, f.name)
        yield f.name
    
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


# ==================== Data Class Tests ====================

class TestRelevanceRegion:
    """Test RelevanceRegion data class"""
    
    def test_creation(self):
        """Test basic creation"""
        region = RelevanceRegion(
            x=10, y=20, width=30, height=40, area=1200,
            centroid=(25, 40), confidence=0.8,
            relevance_score=0.9, is_relevant=True
        )
        
        assert region.x == 10
        assert region.y == 20
        assert region.relevance_score == 0.9
        assert region.is_relevant == True
    
    def test_to_dict(self):
        """Test dictionary conversion"""
        region = RelevanceRegion(
            x=10, y=20, width=30, height=40, area=1200,
            centroid=(25, 40), confidence=0.8,
            relevance_score=0.9, is_relevant=True
        )
        
        d = region.to_dict()
        
        assert d["bbox"]["x"] == 10
        assert d["relevance_score"] == 0.9
        assert d["is_relevant"] == True
    
    def test_to_bbox(self):
        """Test bbox conversion"""
        region = RelevanceRegion(
            x=10, y=20, width=30, height=40, area=1200,
            centroid=(25, 40), confidence=0.8,
            relevance_score=0.9, is_relevant=True
        )
        
        bbox = region.to_bbox()
        assert bbox == (10, 20, 40, 60)


class TestRelevanceMaskResult:
    """Test RelevanceMaskResult data class"""
    
    def test_creation_with_regions(self):
        """Test creation with regions"""
        regions = [
            RelevanceRegion(10, 20, 30, 40, 1200, (25, 40), 0.8, 0.9, True),
            RelevanceRegion(100, 200, 50, 60, 3000, (125, 230), 0.7, 0.3, False)
        ]
        
        mask = np.zeros((480, 640), dtype=np.uint8)
        scores = np.random.rand(480, 640).astype(np.float32)
        
        result = RelevanceMaskResult(
            frame_dimensions=(480, 640),
            relevance_mask=mask,
            relevance_scores=scores,
            regions=regions
        )
        
        assert result.num_relevant == 1
        assert result.num_irrelevant == 1
        assert len(result.relevant_regions) == 1
        assert len(result.irrelevant_regions) == 1
    
    def test_to_dict(self):
        """Test dictionary serialization"""
        regions = [
            RelevanceRegion(10, 20, 30, 40, 1200, (25, 40), 0.8, 0.9, True)
        ]
        
        mask = np.zeros((480, 640), dtype=np.uint8)
        scores = np.zeros((480, 640), dtype=np.float32)
        
        result = RelevanceMaskResult(
            frame_dimensions=(480, 640),
            relevance_mask=mask,
            relevance_scores=scores,
            regions=regions
        )
        
        d = result.to_dict()
        
        assert d["num_relevant"] == 1
        assert d["num_irrelevant"] == 0
        assert len(d["relevant_regions"]) == 1


# ==================== Model Architecture Tests ====================

class TestContrastiveEncoder:
    """Test ContrastiveEncoder module"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = ContrastiveEncoder(feature_dim=128)
        assert model.feature_dim == 128
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = ContrastiveEncoder(feature_dim=128, pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        
        features, projections = model(x)
        
        assert features.shape == (4, 512)
        assert projections.shape == (4, 128)
    
    def test_output_types(self):
        """Test output tensor types"""
        model = ContrastiveEncoder(feature_dim=64, pretrained=False)
        model.eval()
        
        with torch.no_grad():
            x = torch.randn(2, 3, 224, 224)
            features, projections = model(x)
        
        assert isinstance(features, torch.Tensor)
        assert isinstance(projections, torch.Tensor)


class TestRelevanceSegmentationHead:
    """Test RelevanceSegmentationHead module"""
    
    def test_initialization(self):
        """Test initialization"""
        head = RelevanceSegmentationHead(in_channels=512, num_classes=2)
        assert isinstance(head, nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass with correct input shape"""
        head = RelevanceSegmentationHead(in_channels=512, num_classes=2)
        
        # ResNet-18 produces 7x7 feature maps for 224x224 input
        x = torch.randn(4, 512, 7, 7)
        output = head(x)
        
        # Should upsample to 224x224
        assert output.shape == (4, 2, 224, 224)
    
    def test_output_range(self):
        """Test that outputs are logits (not bounded)"""
        head = RelevanceSegmentationHead()
        head.eval()
        
        with torch.no_grad():
            x = torch.randn(2, 512, 7, 7)
            output = head(x)
        
        # Logits can be any value
        assert torch.isfinite(output).all()


class TestRelevanceClassificationHead:
    """Test RelevanceClassificationHead module"""
    
    def test_initialization(self):
        """Test initialization"""
        head = RelevanceClassificationHead(in_features=512, num_classes=2)
        assert isinstance(head, nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass"""
        head = RelevanceClassificationHead(in_features=512, num_classes=2)
        x = torch.randn(4, 512)
        
        output = head(x)
        
        assert output.shape == (4, 2)


class TestRelevanceMaskModel:
    """Test complete RelevanceMaskModel"""
    
    def test_contrastive_mode(self):
        """Test contrastive learning mode"""
        model = RelevanceMaskModel(mode="contrastive", pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        
        output = model(x)
        
        # Should return projections
        assert output.shape == (4, 128)
    
    def test_segmentation_mode(self):
        """Test segmentation mode"""
        model = RelevanceMaskModel(mode="segmentation", pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        
        output = model(x)
        
        # Should return segmentation logits
        assert output.shape == (4, 2, 224, 224)
    
    def test_classification_mode(self):
        """Test classification mode"""
        model = RelevanceMaskModel(mode="classification", pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        
        output = model(x)
        
        # Should return classification logits
        assert output.shape == (4, 2)
    
    def test_both_mode(self):
        """Test both segmentation and classification"""
        model = RelevanceMaskModel(mode="both", pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        
        seg_output, cls_output = model(x)
        
        assert seg_output.shape == (4, 2, 224, 224)
        assert cls_output.shape == (4, 2)
    
    def test_return_features(self):
        """Test returning intermediate features"""
        model = RelevanceMaskModel(mode="both", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        
        seg, cls, features, proj = model(x, return_features=True)
        
        assert features.shape == (2, 512)
        assert proj.shape == (2, 128)


# ==================== Predictor Tests ====================

class TestRelevanceMaskPredictor:
    """Test RelevanceMaskPredictor API"""
    
    def test_initialization_without_model(self):
        """Test initialization without pretrained model"""
        predictor = RelevanceMaskPredictor()
        
        assert predictor.model is not None
        assert predictor.device is not None
    
    def test_initialization_with_model(self, temp_model_path):
        """Test initialization with pretrained model"""
        predictor = RelevanceMaskPredictor(model_path=temp_model_path)
        
        assert predictor.model is not None
    
    def test_predict_no_motion(self, sample_image):
        """Test prediction when no motion detected"""
        predictor = RelevanceMaskPredictor()
        
        # Empty motion result
        motion_result = MotionDetectionResult(
            frame_dimensions=(480, 640),
            motion_detected=False,
            binary_mask=np.zeros((480, 640), dtype=np.uint8),
            motion_regions=[],
            total_motion_area=0,
            confidence=0.0
        )
        
        result = predictor.predict(sample_image, motion_result)
        
        assert result.num_relevant == 0
        assert result.num_irrelevant == 0
        assert result.relevance_mask.shape == (480, 640)
    
    def test_predict_with_motion(self, sample_image, sample_motion_result):
        """Test prediction with motion regions"""
        predictor = RelevanceMaskPredictor()
        
        result = predictor.predict(sample_image, sample_motion_result)
        
        assert result.relevance_mask.shape == (480, 640)
        assert result.relevance_scores.shape == (480, 640)
        assert len(result.regions) > 0
        assert result.processing_time_ms > 0
    
    def test_predict_output_types(self, sample_image, sample_motion_result):
        """Test output data types"""
        predictor = RelevanceMaskPredictor()
        
        result = predictor.predict(sample_image, sample_motion_result)
        
        assert isinstance(result.relevance_mask, np.ndarray)
        assert result.relevance_mask.dtype == np.uint8
        assert isinstance(result.relevance_scores, np.ndarray)
        assert result.relevance_scores.dtype == np.float32
    
    def test_visualize(self, sample_image, sample_motion_result):
        """Test visualization function"""
        predictor = RelevanceMaskPredictor()
        
        result = predictor.predict(sample_image, sample_motion_result)
        vis = predictor.visualize(sample_image, result)
        
        assert vis.shape == sample_image.shape
        assert vis.dtype == np.uint8


# ==================== Utility Function Tests ====================

def test_predict_relevance_mask_convenience(sample_motion_result, temp_model_path):
    """Test convenience function"""
    # Create temporary image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(f.name, img)
        img_path = f.name
    
    try:
        result = predict_relevance_mask(
            img_path,
            sample_motion_result,
            temp_model_path,
            threshold=0.5
        )
        
        assert isinstance(result, RelevanceMaskResult)
    finally:
        Path(img_path).unlink(missing_ok=True)


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests for full pipeline"""
    
    def test_full_pipeline(self, sample_image):
        """Test motion detection + relevance prediction pipeline"""
        # Create temporary images for motion detection
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create sequence of images
            for i in range(3):
                img_path = tmpdir / f"frame_{i}.jpg"
                cv2.imwrite(str(img_path), sample_image)
            
            image_paths = [str(p) for p in sorted(tmpdir.glob("*.jpg"))]
            
            # Step 1: Motion detection
            motion_detector = MotionDetector()
            motion_result = motion_detector.detect_motion(image_paths)
            
            # Step 2: Relevance prediction
            predictor = RelevanceMaskPredictor()
            relevance_result = predictor.predict(sample_image, motion_result)
            
            # Verify pipeline
            assert isinstance(motion_result, MotionDetectionResult)
            assert isinstance(relevance_result, RelevanceMaskResult)
    
    def test_serialization(self, sample_image, sample_motion_result):
        """Test that results can be serialized to JSON"""
        predictor = RelevanceMaskPredictor()
        result = predictor.predict(sample_image, sample_motion_result)
        
        # Convert to dict (should be JSON-serializable)
        result_dict = result.to_dict()
        
        import json
        json_str = json.dumps(result_dict)
        
        assert len(json_str) > 0


# ==================== Edge Case Tests ====================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_image(self):
        """Test with empty image"""
        predictor = RelevanceMaskPredictor()
        
        # Very small image
        tiny_img = np.zeros((10, 10, 3), dtype=np.uint8)
        
        motion_result = MotionDetectionResult(
            frame_dimensions=(10, 10),
            motion_detected=False,
            binary_mask=np.zeros((10, 10), dtype=np.uint8),
            motion_regions=[],
            total_motion_area=0,
            confidence=0.0
        )
        
        result = predictor.predict(tiny_img, motion_result)
        assert result is not None
    
    def test_invalid_model_path(self):
        """Test loading non-existent model"""
        with pytest.raises(FileNotFoundError):
            RelevanceMaskPredictor(model_path="nonexistent_model.pth")
    
    def test_mismatched_dimensions(self, sample_image):
        """Test with mismatched image and mask dimensions"""
        predictor = RelevanceMaskPredictor()
        
        # Motion result with wrong dimensions
        motion_result = MotionDetectionResult(
            frame_dimensions=(100, 100),  # Wrong!
            motion_detected=True,
            binary_mask=np.zeros((100, 100), dtype=np.uint8),
            motion_regions=[],
            total_motion_area=0,
            confidence=0.5
        )
        
        # Should handle gracefully
        result = predictor.predict(sample_image, motion_result)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
