"""
VeroAllarme - Relevance Mask Module (Stage 2)

This module implements a self-supervised learning approach to classify motion regions
as relevant or not relevant. It builds on top of motion_detection.py output.

Key Features:
- Self-supervised contrastive learning (SimCLR-style)
- Motion region classification
- Relevance mask generation
- Integration with motion detection pipeline

Architecture:
1. Feature Extractor: ResNet-18 backbone
2. Projection Head: For contrastive learning
3. Segmentation Head: For pixel-wise relevance scores
4. Classification Head: For region-level relevance scores

Usage:
    from core.relevance_mask import RelevanceMaskPredictor
    from core.motion_detection import MotionDetector
    
    # Detect motion
    detector = MotionDetector()
    motion_result = detector.detect_motion(image_paths)
    
    # Predict relevance
    predictor = RelevanceMaskPredictor(model_path="model.pth")
    relevance_result = predictor.predict(frame, motion_result)
    
    # Access results
    print(f"Relevant regions: {relevance_result.relevant_regions}")
    print(f"Mask shape: {relevance_result.relevance_mask.shape}")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# ==================== Data Classes ====================

@dataclass
class RelevanceRegion:
    """A motion region with relevance classification"""
    x: int
    y: int
    width: int
    height: int
    area: int
    centroid: Tuple[int, int]
    confidence: float  # Motion detection confidence
    relevance_score: float  # Relevance classification score (0-1)
    is_relevant: bool  # True if relevant, False if not
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "bbox": {"x": self.x, "y": self.y, "width": self.width, "height": self.height},
            "area": self.area,
            "centroid": {"x": self.centroid[0], "y": self.centroid[1]},
            "confidence": float(self.confidence),
            "relevance_score": float(self.relevance_score),
            "is_relevant": bool(self.is_relevant)
        }
    
    def to_bbox(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) format"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class RelevanceMaskResult:
    """Complete result from relevance mask prediction"""
    frame_dimensions: Tuple[int, int]  # (height, width)
    relevance_mask: np.ndarray  # Binary mask: 1=relevant, 0=not relevant
    relevance_scores: np.ndarray  # Float scores per pixel (0-1)
    regions: List[RelevanceRegion]
    relevant_regions: List[RelevanceRegion] = field(default_factory=list)
    irrelevant_regions: List[RelevanceRegion] = field(default_factory=list)
    num_relevant: int = 0
    num_irrelevant: int = 0
    processing_time_ms: float = 0.0
    
    def __post_init__(self):
        """Calculate derived fields"""
        self.relevant_regions = [r for r in self.regions if r.is_relevant]
        self.irrelevant_regions = [r for r in self.regions if not r.is_relevant]
        self.num_relevant = len(self.relevant_regions)
        self.num_irrelevant = len(self.irrelevant_regions)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "frame_dimensions": {"height": self.frame_dimensions[0], "width": self.frame_dimensions[1]},
            "num_relevant": self.num_relevant,
            "num_irrelevant": self.num_irrelevant,
            "relevant_regions": [r.to_dict() for r in self.relevant_regions],
            "irrelevant_regions": [r.to_dict() for r in self.irrelevant_regions],
            "processing_time_ms": round(self.processing_time_ms, 2)
        }


# ==================== Model Architecture ====================

class ContrastiveEncoder(nn.Module):
    """
    Self-supervised encoder using contrastive learning.
    Based on SimCLR architecture with ResNet-18 backbone.
    """
    
    def __init__(self, feature_dim: int = 128, pretrained: bool = True):
        """
        Args:
            feature_dim: Dimension of the projection space
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()
        
        # Backbone: ResNet-18
        resnet = models.resnet18(pretrained=pretrained)
        # Feature maps before avgpool (spatial 7x7 for 224x224 input)
        self.backbone_no_pool = nn.Sequential(*list(resnet.children())[:-2])
        # Keep avgpool for classification path
        self.avgpool = resnet.avgpool
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim)
        )
        
        self.feature_dim = feature_dim
    
    def forward(
        self,
        x: torch.Tensor,
        return_spatial: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional spatial feature maps.
        
        Args:
            x: Input tensor (B, C, H, W)
            return_spatial: When True, also return spatial feature maps for segmentation.
        
        Returns:
            pooled_feats: Pooled features (B, 512)
            projections: Projected features for contrastive loss (B, feature_dim)
            spatial_feats (optional): Feature maps before pooling (B, 512, H/32, W/32)
        """
        spatial_feats = self.backbone_no_pool(x)  # e.g., (B, 512, 7, 7)
        pooled = self.avgpool(spatial_feats)  # (B, 512, 1, 1)
        pooled_feats = pooled.view(pooled.size(0), -1)  # (B, 512)
        projections = self.projection_head(pooled_feats)
        if return_spatial:
            return pooled_feats, projections, spatial_feats
        return pooled_feats, projections


class RelevanceSegmentationHead(nn.Module):
    """
    Segmentation head for pixel-wise relevance prediction.
    Uses transposed convolutions to upsample features to original resolution.
    """
    
    def __init__(self, in_channels: int = 512, num_classes: int = 2):
        """
        Args:
            in_channels: Number of input feature channels
            num_classes: Number of output classes (2 for binary: relevant/not relevant)
        """
        super().__init__()
        
        self.decoder = nn.Sequential(
            # Upsample from 7x7 to 14x14
            nn.ConvTranspose2d(in_channels, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Upsample from 14x14 to 28x28
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Upsample from 28x28 to 56x56
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Upsample from 56x56 to 112x112
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Upsample from 112x112 to 224x224
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Final classification layer
            nn.Conv2d(16, num_classes, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Feature maps from backbone (B, 512, 7, 7) for ResNet-18
        
        Returns:
            Segmentation logits (B, num_classes, 224, 224)
        """
        return self.decoder(x)


class RelevanceClassificationHead(nn.Module):
    """
    Classification head for region-level relevance prediction.
    Takes cropped motion regions and predicts relevance score.
    """
    
    def __init__(self, in_features: int = 512, num_classes: int = 2):
        """
        Args:
            in_features: Number of input features from backbone
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Features from backbone (B, 512)
        
        Returns:
            Classification logits (B, num_classes)
        """
        return self.classifier(x)


class RelevanceMaskModel(nn.Module):
    """
    Complete relevance mask model combining:
    1. Contrastive encoder (self-supervised)
    2. Segmentation head (pixel-wise relevance)
    3. Classification head (region-level relevance)
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        num_classes: int = 2,
        pretrained: bool = True,
        mode: str = "both"
    ):
        """
        Args:
            feature_dim: Dimension for contrastive learning projection
            num_classes: Number of classes for classification
            pretrained: Use ImageNet pretrained weights
            mode: "contrastive", "segmentation", "classification", or "both"
        """
        super().__init__()
        
        self.mode = mode
        self.encoder = ContrastiveEncoder(feature_dim, pretrained)
        
        if mode in ["segmentation", "both"]:
            self.segmentation_head = RelevanceSegmentationHead(512, num_classes)
        
        if mode in ["classification", "both"]:
            self.classification_head = RelevanceClassificationHead(512, num_classes)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
            return_features: Whether to return intermediate features
        
        Returns:
            Depends on self.mode:
            - "contrastive": projections
            - "segmentation": segmentation logits
            - "classification": classification logits
            - "both": (segmentation_logits, classification_logits)
        """
        # Get backbone features (need full feature maps for segmentation)
        if self.mode in ["segmentation", "both"]:
            pooled_features, projections, x_features = self.encoder(x, return_spatial=True)
        else:
            # Contrastive/classification only
            pooled_features, projections = self.encoder(x, return_spatial=False)
            x_features = None
        
        # Return based on mode
        if self.mode == "contrastive":
            return projections
        
        elif self.mode == "segmentation":
            seg_logits = self.segmentation_head(x_features)
            if return_features:
                return seg_logits, pooled_features, projections
            return seg_logits
        
        elif self.mode == "classification":
            cls_logits = self.classification_head(pooled_features)
            if return_features:
                return cls_logits, pooled_features, projections
            return cls_logits
        
        else:  # both
            seg_logits = self.segmentation_head(x_features)
            cls_logits = self.classification_head(pooled_features)
            
            if return_features:
                return seg_logits, cls_logits, pooled_features, projections
            return seg_logits, cls_logits


# ==================== Prediction API ====================

class RelevanceMaskPredictor:
    """
    Predictor for generating relevance masks from motion detection results.
    
    This is the main API for inference in the VeroAllarme pipeline.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        relevance_threshold: float = 0.5,
        input_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            model_path: Path to trained model checkpoint (.pth file)
            device: Device to run on ("cuda", "cpu", or None for auto)
            relevance_threshold: Threshold for binary relevance classification (0-1)
            input_size: Input size for model (H, W)
        """
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.relevance_threshold = relevance_threshold
        self.input_size = input_size
        
        # Initialize model
        self.model = RelevanceMaskModel(mode="both", pretrained=False)
        
        # Load checkpoint if provided
        if model_path is not None:
            self.load_model(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load model weights from checkpoint"""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"✓ Loaded model from {model_path}")
    
    def predict(
        self,
        frame: Union[str, Path, np.ndarray],
        motion_result,  # MotionDetectionResult from motion_detection.py
        visualize: bool = False
    ) -> RelevanceMaskResult:
        """
        Predict relevance mask for a frame with detected motion regions.
        
        Args:
            frame: Input frame (path or numpy array)
            motion_result: Result from MotionDetector.detect_motion()
            visualize: Whether to return visualization-ready outputs
        
        Returns:
            RelevanceMaskResult with classified regions and mask
        """
        import time
        start_time = time.time()
        
        # Load frame if path provided
        if isinstance(frame, (str, Path)):
            frame = cv2.imread(str(frame))
            if frame is None:
                raise ValueError(f"Failed to load frame: {frame}")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # If no motion detected, return empty result
        if not motion_result.motion_detected or len(motion_result.motion_regions) == 0:
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            empty_scores = np.zeros((h, w), dtype=np.float32)
            
            return RelevanceMaskResult(
                frame_dimensions=(h, w),
                relevance_mask=empty_mask,
                relevance_scores=empty_scores,
                regions=[],
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Process full frame for segmentation
        frame_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            seg_logits, cls_logits = self.model(frame_tensor)
            
            # Get segmentation mask
            seg_probs = F.softmax(seg_logits, dim=1)  # (1, 2, 224, 224)
            relevance_scores_small = seg_probs[0, 1].cpu().numpy()  # (224, 224)
            
            # Resize to original frame size
            relevance_scores = cv2.resize(relevance_scores_small, (w, h))
            relevance_mask = (relevance_scores > self.relevance_threshold).astype(np.uint8)
        
        # Classify each motion region
        relevance_regions = []
        
        for motion_region in motion_result.motion_regions:
            # Extract region crop
            x1, y1 = motion_region.x, motion_region.y
            x2, y2 = x1 + motion_region.width, y1 + motion_region.height
            
            # Ensure bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = frame_rgb[y1:y2, x1:x2]
            
            # Classify crop
            crop_tensor = self.transform(crop).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                _, crop_cls_logits = self.model(crop_tensor)
                crop_probs = F.softmax(crop_cls_logits, dim=1)
                relevance_score = crop_probs[0, 1].item()
            
            is_relevant = relevance_score > self.relevance_threshold
            
            relevance_regions.append(RelevanceRegion(
                x=motion_region.x,
                y=motion_region.y,
                width=motion_region.width,
                height=motion_region.height,
                area=motion_region.area,
                centroid=motion_region.centroid,
                confidence=1.0,  # Motion detection confidence
                relevance_score=relevance_score,
                is_relevant=is_relevant
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return RelevanceMaskResult(
            frame_dimensions=(h, w),
            relevance_mask=relevance_mask,
            relevance_scores=relevance_scores,
            regions=relevance_regions,
            processing_time_ms=processing_time
        )
    
    def visualize(
        self,
        frame: np.ndarray,
        result: RelevanceMaskResult,
        output_path: Optional[Union[str, Path]] = None,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Visualize relevance mask and classified regions.
        
        Args:
            frame: Original frame (BGR)
            result: RelevanceMaskResult from predict()
            output_path: Optional path to save visualization
            alpha: Transparency for mask overlay
        
        Returns:
            Visualization image (BGR)
        """
        vis = frame.copy()
        h, w = vis.shape[:2]
        
        # Create colored mask overlay
        mask_overlay = np.zeros_like(vis)
        mask_overlay[:, :, 1] = result.relevance_mask * 255  # Green for relevant
        
        # Blend with original
        vis = cv2.addWeighted(vis, 1 - alpha, mask_overlay, alpha, 0)
        
        # Draw bounding boxes
        for region in result.relevant_regions:
            x1, y1, x2, y2 = region.to_bbox()
            # Green box for relevant
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis, region.centroid, 4, (0, 255, 255), -1)
            
            # Label
            label = f"REL: {region.relevance_score:.2f}"
            cv2.putText(vis, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for region in result.irrelevant_regions:
            x1, y1, x2, y2 = region.to_bbox()
            # Red box for irrelevant
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            label = f"IRR: {region.relevance_score:.2f}"
            cv2.putText(vis, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add summary text
        summary = f"Relevant: {result.num_relevant} | Irrelevant: {result.num_irrelevant}"
        cv2.putText(vis, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if output_path:
            cv2.imwrite(str(output_path), vis)
            print(f"✓ Saved visualization to {output_path}")
        
        return vis


# ==================== Utility Functions ====================

def predict_relevance_mask(
    frame_path: Union[str, Path],
    motion_result,
    model_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    threshold: float = 0.5
) -> RelevanceMaskResult:
    """
    Convenience function for single-frame prediction.
    
    Args:
        frame_path: Path to input frame
        motion_result: MotionDetectionResult from motion detection
        model_path: Path to trained model checkpoint
        output_path: Optional path to save visualization
        threshold: Relevance threshold
    
    Returns:
        RelevanceMaskResult
    """
    predictor = RelevanceMaskPredictor(
        model_path=model_path,
        relevance_threshold=threshold
    )
    
    result = predictor.predict(frame_path, motion_result)
    
    if output_path:
        frame = cv2.imread(str(frame_path))
        predictor.visualize(frame, result, output_path)
    
    return result


if __name__ == "__main__":
    print("VeroAllarme - Relevance Mask Module")
    print("=" * 60)
    print("\nThis module provides relevance masking for motion regions.")
    print("Use RelevanceMaskPredictor for inference.")
    print("\nExample usage:")
    print("  from core.relevance_mask import RelevanceMaskPredictor")
    print("  predictor = RelevanceMaskPredictor(model_path='model.pth')")
    print("  result = predictor.predict(frame, motion_result)")
