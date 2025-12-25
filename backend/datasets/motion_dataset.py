"""
VeroAllarme - Dataset Loaders for Relevance Mask Training

This module provides dataset loaders for self-supervised contrastive learning
from motion detection sequences.

Dataset Types:
1. ContrastiveMotionDataset: For self-supervised learning with triplets
2. MotionRegionDataset: For supervised region classification
3. SegmentationDataset: For pixel-wise segmentation training

Usage:
    from datasets.motion_dataset import ContrastiveMotionDataset
    
    dataset = ContrastiveMotionDataset(
        data_dir="data/training/camera-events/Factory",
        motion_detector=MotionDetector()
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

from pathlib import Path
from typing import List, Tuple, Optional, Callable
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.motion_detection import MotionDetector


# ==================== Augmentation Functions ====================

class ContrastiveAugmentation:
    """
    Augmentation pipeline for contrastive learning.
    Creates two different augmented views of the same image.
    """
    
    def __init__(self, size: Tuple[int, int] = (224, 224)):
        """
        Args:
            size: Target size (H, W)
        """
        # Strong augmentation for contrastive learning
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Weak augmentation (for some contrastive methods)
        self.weak_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image: np.ndarray, weak: bool = False) -> torch.Tensor:
        """
        Apply augmentation
        
        Args:
            image: Input image (H, W, C) in RGB format
            weak: If True, use weak augmentation
        
        Returns:
            Augmented tensor
        """
        if weak:
            return self.weak_transform(image)
        return self.transform(image)


# ==================== Dataset Classes ====================

class ContrastiveMotionDataset(Dataset):
    """
    Dataset for self-supervised contrastive learning from motion sequences.
    
    Each sample consists of:
    - Anchor: A frame from a motion sequence
    - Positive: A different frame from the SAME sequence (temporal consistency)
    - Negative: A frame from a DIFFERENT sequence
    
    This creates meaningful contrastive pairs without requiring labels.
    """
    
    def __init__(
        self,
        data_dir: str,
        motion_detector: Optional[MotionDetector] = None,
        transform: Optional[Callable] = None,
        min_motion_area: int = 500,
        max_sequences: Optional[int] = None,
        cache_detections: bool = True
    ):
        """
        Args:
            data_dir: Root directory containing event subdirectories
            motion_detector: MotionDetector instance (or will create one)
            transform: Augmentation transform
            min_motion_area: Minimum motion area to include sequence
            max_sequences: Maximum number of sequences to load (None = all)
            cache_detections: Cache motion detection results
        """
        self.data_dir = Path(data_dir)
        self.min_motion_area = min_motion_area
        self.cache_detections = cache_detections
        
        # Initialize motion detector
        if motion_detector is None:
            self.motion_detector = MotionDetector()
        else:
            self.motion_detector = motion_detector
        
        # Initialize transform
        if transform is None:
            self.transform = ContrastiveAugmentation()
        else:
            self.transform = transform
        
        # Load sequences
        self.sequences = self._load_sequences(max_sequences)
        self.detection_cache = {} if cache_detections else None
        
        print(f"✓ Loaded {len(self.sequences)} motion sequences from {data_dir}")
    
    def _load_sequences(self, max_sequences: Optional[int]) -> List[dict]:
        """Load all motion event sequences"""
        sequences = []
        
        # Find all event directories
        event_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        if max_sequences:
            event_dirs = event_dirs[:max_sequences]
        
        for event_dir in event_dirs:
            # Find all images in event
            images = sorted([str(p) for p in event_dir.glob("*.jpg")])
            
            if len(images) < 2:
                continue  # Need at least 2 frames for positive pairs
            
            sequences.append({
                "event_id": event_dir.name,
                "images": images,
                "num_frames": len(images)
            })
        
        return sequences
    
    def _get_motion_result(self, images: List[str]):
        """Get or compute motion detection result"""
        cache_key = tuple(images)
        
        if self.cache_detections and cache_key in self.detection_cache:
            return self.detection_cache[cache_key]
        
        # Compute motion detection
        result = self.motion_detector.detect_motion(images)
        
        if self.cache_detections:
            self.detection_cache[cache_key] = result
        
        return result
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get contrastive triplet: (anchor, positive, negative)
        
        Returns:
            anchor: Augmented view 1 of frame from sequence[idx]
            positive: Augmented view 2 of different frame from sequence[idx]
            negative: Augmented view of frame from different sequence
        """
        # Get anchor sequence
        anchor_seq = self.sequences[idx]
        
        # Select two random frames from anchor sequence
        anchor_idx = random.randint(0, anchor_seq["num_frames"] - 1)
        positive_idx = random.randint(0, anchor_seq["num_frames"] - 1)
        
        # Ensure positive is different from anchor
        while positive_idx == anchor_idx and anchor_seq["num_frames"] > 1:
            positive_idx = random.randint(0, anchor_seq["num_frames"] - 1)
        
        # Select random negative sequence (different from anchor)
        negative_seq_idx = random.randint(0, len(self.sequences) - 1)
        while negative_seq_idx == idx and len(self.sequences) > 1:
            negative_seq_idx = random.randint(0, len(self.sequences) - 1)
        
        negative_seq = self.sequences[negative_seq_idx]
        negative_idx = random.randint(0, negative_seq["num_frames"] - 1)
        
        # Load images
        anchor_img = cv2.imread(anchor_seq["images"][anchor_idx])
        anchor_img = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB)
        
        positive_img = cv2.imread(anchor_seq["images"][positive_idx])
        positive_img = cv2.cvtColor(positive_img, cv2.COLOR_BGR2RGB)
        
        negative_img = cv2.imread(negative_seq["images"][negative_idx])
        negative_img = cv2.cvtColor(negative_img, cv2.COLOR_BGR2RGB)
        
        # Populate detection cache for anchor sequence when enabled
        if self.cache_detections:
            _ = self._get_motion_result(anchor_seq["images"])

        # Apply augmentations
        anchor_tensor = self.transform(anchor_img)
        positive_tensor = self.transform(positive_img)
        negative_tensor = self.transform(negative_img)
        
        return anchor_tensor, positive_tensor, negative_tensor


class MotionRegionDataset(Dataset):
    """
    Dataset for supervised region classification.
    Extracts motion region crops and classifies them.
    
    Requires labels: relevant (1) or not relevant (0)
    """
    
    def __init__(
        self,
        data_dir: str,
        motion_detector: Optional[MotionDetector] = None,
        labels_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        min_region_size: int = 32,
        max_region_size: int = 512
    ):
        """
        Args:
            data_dir: Root directory containing event subdirectories
            motion_detector: MotionDetector instance
            labels_file: Optional JSON file with region labels
            transform: Augmentation transform
            min_region_size: Minimum region dimension (width or height)
            max_region_size: Maximum region dimension
        """
        self.data_dir = Path(data_dir)
        self.min_region_size = min_region_size
        self.max_region_size = max_region_size
        
        # Initialize motion detector
        if motion_detector is None:
            self.motion_detector = MotionDetector()
        else:
            self.motion_detector = motion_detector
        
        # Initialize transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Load labels if provided
        self.labels = self._load_labels(labels_file) if labels_file else {}
        
        # Extract all regions
        self.regions = self._extract_regions()
        
        print(f"✓ Extracted {len(self.regions)} motion regions from {data_dir}")
    
    def _load_labels(self, labels_file: str) -> dict:
        """Load region labels from JSON"""
        import json
        
        labels_path = Path(labels_file)
        if not labels_path.exists():
            print(f"⚠ Labels file not found: {labels_file}")
            return {}
        
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        
        print(f"✓ Loaded {len(labels)} labeled sequences")
        return labels
    
    def _extract_regions(self) -> List[dict]:
        """Extract all motion regions from sequences"""
        regions = []
        
        event_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for event_dir in event_dirs:
            images = sorted([str(p) for p in event_dir.glob("*.jpg")])
            
            if len(images) < 2:
                continue
            
            # Detect motion
            try:
                result = self.motion_detector.detect_motion(images)
            except Exception as e:
                print(f"⚠ Error processing {event_dir.name}: {e}")
                continue
            
            if not result.motion_detected:
                continue
            
            # Load middle frame for region extraction
            middle_idx = len(images) // 2
            frame = cv2.imread(images[middle_idx])
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract each region
            for region in result.motion_regions:
                # Filter by size
                if (region.width < self.min_region_size or 
                    region.height < self.min_region_size or
                    region.width > self.max_region_size or 
                    region.height > self.max_region_size):
                    continue
                
                x1, y1 = region.x, region.y
                x2, y2 = x1 + region.width, y1 + region.height
                
                # Extract crop
                crop = frame_rgb[y1:y2, x1:x2]
                
                # Get label if available
                event_id = event_dir.name
                label = self.labels.get(event_id, -1)  # -1 = unlabeled
                
                regions.append({
                    "crop": crop,
                    "event_id": event_id,
                    "region": region,
                    "label": label
                })
        
        return regions
    
    def __len__(self) -> int:
        return len(self.regions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get region crop and label
        
        Returns:
            crop_tensor: Transformed crop image
            label: Class label (0=not relevant, 1=relevant, -1=unlabeled)
        """
        region_data = self.regions[idx]
        crop = region_data["crop"]
        label = region_data["label"]
        
        # Apply transform
        crop_tensor = self.transform(crop)
        
        return crop_tensor, label


class SegmentationDataset(Dataset):
    """
    Dataset for pixel-wise segmentation training.
    Generates pseudo-labels from motion detection masks.
    """
    
    def __init__(
        self,
        data_dir: str,
        motion_detector: Optional[MotionDetector] = None,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224),
        max_sequences: Optional[int] = None
    ):
        """
        Args:
            data_dir: Root directory containing event subdirectories
            motion_detector: MotionDetector instance
            transform: Image transform
            target_size: Target size (H, W) for training
            max_sequences: Maximum sequences to load
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        
        # Initialize motion detector
        if motion_detector is None:
            self.motion_detector = MotionDetector()
        else:
            self.motion_detector = motion_detector
        
        # Image transform
        if transform is None:
            self.img_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = transform
        
        # Mask transform (no normalization)
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        
        # Load sequences
        self.samples = self._load_samples(max_sequences)
        
        print(f"✓ Loaded {len(self.samples)} samples for segmentation training")
    
    def _load_samples(self, max_sequences: Optional[int]) -> List[dict]:
        """Load image-mask pairs"""
        samples = []
        
        event_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        if max_sequences:
            event_dirs = event_dirs[:max_sequences]
        
        for event_dir in event_dirs:
            images = sorted([str(p) for p in event_dir.glob("*.jpg")])
            
            if len(images) < 2:
                continue
            
            # Detect motion
            try:
                result = self.motion_detector.detect_motion(images)
            except Exception as e:
                print(f"⚠ Error processing {event_dir.name}: {e}")
                continue
            
            if not result.motion_detected:
                continue
            
            # Use middle frame and its mask
            middle_idx = len(images) // 2
            frame_path = images[middle_idx]
            
            # Convert motion_mask to binary (0 or 1)
            binary_mask = (result.motion_mask > 0).astype(np.uint8)
            
            samples.append({
                "image_path": frame_path,
                "mask": binary_mask
            })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get image and segmentation mask
        
        Returns:
            image_tensor: Transformed image (3, H, W)
            mask_tensor: Binary mask (1, H, W)
        """
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get mask
        mask = sample["mask"]
        
        # Apply transforms
        image_tensor = self.img_transform(image)
        mask_tensor = self.mask_transform(mask)
        
        return image_tensor, mask_tensor


# ==================== Utility Functions ====================

def create_contrastive_dataloader(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Convenience function to create contrastive learning dataloader.
    
    Args:
        data_dir: Data directory path
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle
        **kwargs: Additional arguments for ContrastiveMotionDataset
    
    Returns:
        DataLoader instance
    """
    dataset = ContrastiveMotionDataset(data_dir, **kwargs)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


if __name__ == "__main__":
    print("VeroAllarme - Motion Dataset Loaders")
    print("=" * 60)
    
    # Test dataset
    data_dir = "../../data/training/camera-events/Factory"
    
    if Path(data_dir).exists():
        print(f"\nTesting ContrastiveMotionDataset on {data_dir}...")
        
        dataset = ContrastiveMotionDataset(data_dir, max_sequences=5)
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            anchor, positive, negative = dataset[0]
            print(f"Sample shapes:")
            print(f"  Anchor: {anchor.shape}")
            print(f"  Positive: {positive.shape}")
            print(f"  Negative: {negative.shape}")
    else:
        print(f"⚠ Data directory not found: {data_dir}")
