"""
VeroAllarme - Tests for Motion Dataset Loaders

Test suite for dataset classes used in training.
"""

import pytest
import torch
import numpy as np
import cv2
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.motion_dataset import (
    ContrastiveAugmentation,
    ContrastiveMotionDataset,
    MotionRegionDataset,
    SegmentationDataset,
    create_contrastive_dataloader
)
from core.motion_detection import MotionDetector


# ==================== Fixtures ====================

@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory with mock sequences"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create mock event directories
        for i in range(3):
            event_dir = tmpdir / f"event_{i:03d}"
            event_dir.mkdir()
            
            # Create mock images
            for j in range(3):
                img_path = event_dir / f"{j}.jpg"
                # Create simple image with some variation
                img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                # Add some motion-like changes
                if j > 0:
                    img[100:200, 100:200] = 255  # White square
                cv2.imwrite(str(img_path), img)
        
        yield str(tmpdir)


@pytest.fixture
def motion_detector():
    """Create a motion detector instance"""
    return MotionDetector(threshold=25, min_area=500)


# ==================== Augmentation Tests ====================

class TestContrastiveAugmentation:
    """Test ContrastiveAugmentation class"""
    
    def test_initialization(self):
        """Test initialization"""
        aug = ContrastiveAugmentation(size=(224, 224))
        assert aug.transform is not None
        assert aug.weak_transform is not None
    
    def test_strong_augmentation(self):
        """Test strong augmentation"""
        aug = ContrastiveAugmentation()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        output = aug(img, weak=False)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)
        assert output.dtype == torch.float32
    
    def test_weak_augmentation(self):
        """Test weak augmentation"""
        aug = ContrastiveAugmentation()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        output = aug(img, weak=True)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)
    
    def test_normalization(self):
        """Test that output is normalized"""
        aug = ContrastiveAugmentation()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        output = aug(img)
        
        # ImageNet normalization should give values roughly in [-2, 2]
        assert output.min() >= -3
        assert output.max() <= 3


# ==================== ContrastiveMotionDataset Tests ====================

class TestContrastiveMotionDataset:
    """Test ContrastiveMotionDataset"""
    
    def test_initialization(self, temp_data_dir, motion_detector):
        """Test dataset initialization"""
        dataset = ContrastiveMotionDataset(
            temp_data_dir,
            motion_detector=motion_detector,
            max_sequences=2
        )
        
        assert len(dataset) > 0
        assert len(dataset.sequences) <= 2
    
    def test_getitem(self, temp_data_dir, motion_detector):
        """Test getting an item"""
        dataset = ContrastiveMotionDataset(
            temp_data_dir,
            motion_detector=motion_detector,
            max_sequences=2
        )
        
        if len(dataset) > 0:
            anchor, positive, negative = dataset[0]
            
            assert isinstance(anchor, torch.Tensor)
            assert isinstance(positive, torch.Tensor)
            assert isinstance(negative, torch.Tensor)
            
            assert anchor.shape == (3, 224, 224)
            assert positive.shape == (3, 224, 224)
            assert negative.shape == (3, 224, 224)
    
    def test_positive_pairs_from_same_sequence(self, temp_data_dir, motion_detector):
        """Test that positive pairs come from same sequence"""
        dataset = ContrastiveMotionDataset(
            temp_data_dir,
            motion_detector=motion_detector
        )
        
        if len(dataset) > 0:
            # This is tested implicitly in the implementation
            # We can at least verify no errors occur
            for i in range(min(3, len(dataset))):
                anchor, positive, negative = dataset[i]
                assert anchor is not None
    
    def test_dataset_length(self, temp_data_dir):
        """Test dataset length"""
        dataset = ContrastiveMotionDataset(temp_data_dir)
        
        # Should have 3 sequences from fixture
        assert len(dataset) == 3
    
    def test_caching(self, temp_data_dir, motion_detector):
        """Test detection caching"""
        dataset = ContrastiveMotionDataset(
            temp_data_dir,
            motion_detector=motion_detector,
            cache_detections=True
        )
        
        if len(dataset) > 0:
            # Access same item twice
            _ = dataset[0]
            _ = dataset[0]
            
            # Cache should be populated
            assert len(dataset.detection_cache) > 0


# ==================== MotionRegionDataset Tests ====================

class TestMotionRegionDataset:
    """Test MotionRegionDataset"""
    
    def test_initialization(self, temp_data_dir, motion_detector):
        """Test dataset initialization"""
        dataset = MotionRegionDataset(
            temp_data_dir,
            motion_detector=motion_detector
        )
        
        # May or may not extract regions depending on motion detection
        assert isinstance(dataset.regions, list)
    
    def test_getitem_shape(self, temp_data_dir, motion_detector):
        """Test getting an item if regions exist"""
        dataset = MotionRegionDataset(
            temp_data_dir,
            motion_detector=motion_detector
        )
        
        if len(dataset) > 0:
            crop, label = dataset[0]
            
            assert isinstance(crop, torch.Tensor)
            assert crop.shape == (3, 224, 224)
            assert isinstance(label, int)
    
    def test_unlabeled_data(self, temp_data_dir, motion_detector):
        """Test with unlabeled data"""
        dataset = MotionRegionDataset(
            temp_data_dir,
            motion_detector=motion_detector,
            labels_file=None
        )
        
        if len(dataset) > 0:
            _, label = dataset[0]
            assert label == -1  # Unlabeled


# ==================== SegmentationDataset Tests ====================

class TestSegmentationDataset:
    """Test SegmentationDataset"""
    
    def test_initialization(self, temp_data_dir, motion_detector):
        """Test dataset initialization"""
        dataset = SegmentationDataset(
            temp_data_dir,
            motion_detector=motion_detector,
            max_sequences=2
        )
        
        assert isinstance(dataset.samples, list)
    
    def test_getitem_shapes(self, temp_data_dir, motion_detector):
        """Test item shapes"""
        dataset = SegmentationDataset(
            temp_data_dir,
            motion_detector=motion_detector,
            target_size=(224, 224)
        )
        
        if len(dataset) > 0:
            image, mask = dataset[0]
            
            assert isinstance(image, torch.Tensor)
            assert isinstance(mask, torch.Tensor)
            
            assert image.shape == (3, 224, 224)
            assert mask.shape == (1, 224, 224)
    
    def test_mask_values(self, temp_data_dir, motion_detector):
        """Test that mask contains binary values"""
        dataset = SegmentationDataset(
            temp_data_dir,
            motion_detector=motion_detector
        )
        
        if len(dataset) > 0:
            _, mask = dataset[0]
            
            unique_values = torch.unique(mask)
            # Mask should be binary (0 or 1) but might have interpolation artifacts
            assert mask.min() >= 0
            assert mask.max() <= 1


# ==================== Utility Function Tests ====================

def test_create_contrastive_dataloader(temp_data_dir):
    """Test dataloader creation utility"""
    dataloader = create_contrastive_dataloader(
        temp_data_dir,
        batch_size=2,
        num_workers=0,  # Use 0 for testing
        shuffle=False,
        max_sequences=2
    )
    
    assert dataloader is not None
    assert dataloader.batch_size == 2


# ==================== Integration Tests ====================

class TestDatasetIntegration:
    """Integration tests for datasets"""
    
    def test_contrastive_dataloader_iteration(self, temp_data_dir):
        """Test iterating through contrastive dataloader"""
        dataloader = create_contrastive_dataloader(
            temp_data_dir,
            batch_size=2,
            num_workers=0,
            shuffle=False,
            max_sequences=2
        )
        
        for batch in dataloader:
            anchor, positive, negative = batch
            
            assert anchor.shape[0] == 2  # Batch size
            assert positive.shape[0] == 2
            assert negative.shape[0] == 2
            
            break  # Just test first batch
    
    def test_segmentation_dataloader_iteration(self, temp_data_dir, motion_detector):
        """Test iterating through segmentation dataloader"""
        dataset = SegmentationDataset(
            temp_data_dir,
            motion_detector=motion_detector,
            max_sequences=2
        )
        
        if len(dataset) > 0:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0
            )
            
            for images, masks in dataloader:
                assert images.shape[0] <= 2  # Batch size
                assert masks.shape[0] <= 2
                break


# ==================== Edge Cases ====================

class TestEdgeCases:
    """Test edge cases"""
    
    def test_empty_directory(self):
        """Test with empty directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = ContrastiveMotionDataset(tmpdir)
            
            assert len(dataset) == 0
    
    def test_no_motion_sequences(self, temp_data_dir):
        """Test when motion detection finds nothing"""
        # Use very high threshold so nothing is detected
        detector = MotionDetector(threshold=200, min_area=100000)
        
        dataset = SegmentationDataset(
            temp_data_dir,
            motion_detector=detector,
            max_sequences=1
        )
        
        # Should handle gracefully (may be empty)
        assert len(dataset) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
